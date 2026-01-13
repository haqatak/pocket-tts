import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Optional
import copy

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.config import Config
from training.data_prep import SSCProcessor

logger = logging.getLogger(__name__)

class SSCDataset(Dataset):
    def __init__(self, data_list, processor):
        self.data_list = data_list # List of (audio_path, text)
        self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_path, text = self.data_list[idx]
        item = self.processor.create_dataset_item(audio_path, text)
        return item

def train_one_epoch(model: TTSModel, dataloader, optimizer, device, head_batch_multiplier=8):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        prompt_latents = batch['prompt_latent'].to(device) # [B, C, T_prompt]
        target_latents = batch['target_latent'].to(device) # [B, C, T_target]
        text_tokens = model.flow_lm.conditioner.prepare(batch['text'][0]).tokens.to(device) # Batch size 1 assumption for now

        # Get audio conditioning from prompt
        # [B, C, T] -> [B, T, C]
        prompt_latents_t = prompt_latents.transpose(1, 2)
        audio_conditioning = torch.nn.functional.linear(prompt_latents_t, model.flow_lm.speaker_proj_weight)

        # Prepare inputs for backbone
        # Text embeddings
        # The input to _get_condition should be TokenizedText, but here we extracted tokens.
        # But wait, TokenizedText is a named tuple or similar.
        # Check LUTConditioner._get_condition signature.
        # It expects `inputs: TokenizedText`.
        # And TokenizedText wraps the tensor.
        from pocket_tts.conditioners.base import TokenizedText
        text_embeddings = model.flow_lm.conditioner._get_condition(TokenizedText(text_tokens))
        # Concatenate audio conditioning to text embeddings
        text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)

        # History Injection (Training Logic)
        # We want to predict target_latents.
        # target_latents is [B, C, T]. Transpose to [B, T, C].
        targets = target_latents.transpose(1, 2)

        # Add noise to history (regularization)
        history = targets.clone()
        history_noise_std = 0.1
        history = history + torch.randn_like(history) * history_noise_std

        # Input to backbone is projected history
        history_emb = model.flow_lm.input_linear(history)

        # Run backbone
        # We need to initialize model state properly using init_states.
        # FlowLM's backbone (Transformer) is stateful (KV cache etc).
        # We need to compute the max sequence length we expect.
        # history length is T.
        # text_embeddings length is K.
        # Total length needed is K + T.

        from pocket_tts.modules.stateful_module import init_states

        # We need to set names on modules if not set (init_states sets them).
        # But we are in a training loop.
        # `init_states` iterates named_modules and calls init_state.

        batch_size = history_emb.shape[0]
        # Max sequence length.
        seq_len = history_emb.shape[1] + text_embeddings.shape[1]

        model_state = init_states(model.flow_lm, batch_size=batch_size, sequence_length=seq_len + 10)

        context = model.flow_lm.backbone(history_emb, text_embeddings, history, model_state)
        # context: [B, T, D]

        # Head Batch Multiplier
        # Repeat context N times
        N = head_batch_multiplier
        B, T, D = context.shape

        context_flat = context.view(-1, D) # [B*T, D]
        context_rep = context_flat.repeat_interleave(N, dim=0) # [B*T*N, D]

        targets_flat = targets.view(-1, model.flow_lm.ldim)
        targets_rep = targets_flat.repeat_interleave(N, dim=0) # [B*T*N, L]

        # Sample t and noise for Consistency/Flow Matching
        # We model the flow from Data (t=0) to Noise (t=1).
        # x_0 = targets
        # x_1 = noise
        # x_t = (1-t) x_0 + t x_1
        # vector field v_t = x_1 - x_0

        # Sample t uniform [0, 1]
        t = torch.rand(B*T*N, 1, device=device)
        noise = torch.randn_like(targets_rep)

        x_0 = targets_rep
        x_1 = noise
        x_t = (1-t) * x_0 + t * x_1

        # Note: SimpleMLPAdaLN expects s (source time) and t (target time)
        # The prompt says "Consistency Loss".
        # If we use `lsd_decode`, it takes v_t(s, t, x).
        # `FlowLMModel` calls `flow_net` via `conditioned_flow = partial(self.flow_net, transformer_out)`.
        # And `lsd_decode` calls `v_t(s, t, current)`.

        # So flow_net signature is (cond, s, t, x).
        # In LSD paper, s is current time, t is next time (s + ds).
        # But `SimpleMLPAdaLN` treats them as source and target times for flow?
        # Let's check `mlp.py` -> `forward(c, s, t, x)`.

        # We want to train the vector field v(x, t).
        # In this formulation, we are predicting the flow at time t.
        # So we can pass s=t, t=t? Or does it expect a step?
        # Usually vector field is v(x, t).
        # LSD calls `v_t(s, t, current)`.
        # s=current_time, t=next_time.
        # If we are training the vector field to be constant straight line (x1-x0),
        # then v does not depend on time for the straight path, but the network might.

        # Let's set s = t, and t_arg = t.
        # Or maybe s is t_current and t is t_target?
        # In `lsd_decode`: s = i/N, t = (i+1)/N.
        # So s and t are close.
        # Let's use s=t, t=t for training the instantaneous velocity at t.

        s_in = t
        t_in = t

        prediction = model.flow_lm.flow_net(context_rep, s_in, t_in, x_t)

        v_target = x_1 - x_0

        loss = nn.MSELoss()(prediction, v_target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
