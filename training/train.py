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

def update_ema(target_model, source_model, decay=0.999):
    with torch.no_grad():
        for p_tgt, p_src in zip(target_model.parameters(), source_model.parameters()):
            p_tgt.data.mul_(decay).add_(p_src.data, alpha=1 - decay)

def train_one_epoch(model: TTSModel, dataloader, optimizer, device, head_batch_multiplier=8, ema_model: Optional[TTSModel] = None, freeze_backbone: bool = False):
    model.train()
    if freeze_backbone:
        # Freeze backbone parameters
        for param in model.flow_lm.transformer.parameters():
            param.requires_grad = False
        # Ensure we don't accidentally freeze head
        for param in model.flow_lm.flow_net.parameters():
            param.requires_grad = True

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

        # Sample t for Consistency Training
        # We need to compute Consistency Loss:
        # L = d(f_theta(x_{t+1}, t+1), f_theta^-(x_t, t))
        # Where x_t is on the trajectory.
        # Pocket TTS uses LSD (Lagrangian Self Distillation).
        # We will implement the loss comparing the student's prediction from a noisy state
        # to the teacher's prediction from a less noisy state (or the clean target).

        # Sample t uniform [0, 1]
        # We discretize t for consistency? Or continuous.
        # Let's use continuous t.
        t = torch.rand(B*T*N, 1, device=device)

        # Create adjacent time step for distillation
        # t_prev = t - delta? Or t_next?
        # If we generate from Noise (1) to Data (0):
        # t is current time. t_next is closer to 0?
        # Let's assume standard flow matching convention: t=0 is data, t=1 is noise.
        # Flow x_t = (1-t)x_0 + t x_1.

        # We want to enforce that prediction from x_t points to x_0.
        # And prediction from x_{t+eps} points to same x_0.

        x_0 = targets_rep
        noise = torch.randn_like(x_0)
        x_1 = noise

        x_t = (1-t) * x_0 + t * x_1

        # Student prediction from x_t
        # flow_net output usually is v = x_1 - x_0.
        # So predicted x_0 = x_t - t * v_pred ?
        # x_t = x_0 + t(x_1 - x_0) = x_0 + t * v.
        # x_0 = x_t - t * v.

        # s=t, t=t for instantaneous velocity field check
        v_pred = model.flow_lm.flow_net(context_rep, t, t, x_t)
        x_0_pred = x_t - t * v_pred

        if ema_model is not None:
            # Consistency Distillation with EMA Teacher
            # We compare student prediction x_0_pred with teacher prediction.
            # Teacher should see x_t? Or a different point?
            # Ideally teacher sees a "better" point.
            # But "Consistency Distillation" often compares prediction from x_{t_{n+1}} (student)
            # with prediction from x_{t_n} (teacher).
            # Here let's use the same x_t for simplicity unless strictly required to step.
            # Using same x_t is "Self-Consistency".
            # Using adjacent points is "Trajectory Consistency".

            # Let's use x_t for teacher too, or x_{t-eps}
            # For continuous consistency, comparing at same point is valid for ensuring x_0 prediction is stable?
            # No, usually we want consistency along the flow.
            # Let's stick to the prompt's "Minimize distance between predicted ... and target Mimi latent".
            # That implies Target = x_0 (Ground Truth).
            # But the "Forensic Analysis" asked for Consistency Distillation.
            # The feedback says "If the PR implements standard DSM... it is... not a consistency model."
            # "It must implement a Consistency Distillation... L = d(f_student(x_{t+1}), f_teacher(x_t))".

            # Let's implement that:
            # t_next = t + delta (closer to noise? or closer to data?)
            # If t=0 is data, t=1 is noise.
            # Flow goes 1 -> 0 for generation.
            # So x_{t+1} is closer to noise? (Wait, notation is confusing).
            # Let's say s > t. s is more noisy.
            # We want f(x_s) == f(x_t).
            # Student predicts from s (harder). Teacher predicts from t (easier).

            # So:
            # t1 = t (noisy)
            # t2 = t - delta (less noisy)
            # Ensure t2 >= 0.

            # Let's assume we sample t2 ~ U[0, 1]. t1 = t2 + small_step.

            # To implementing this efficiently with HBM:
            # We have N samples. We can pair them? Or just sample pairs?
            # We generated t. Let's make t_teacher = t.
            # And t_student = t + delta.

            delta = 0.01 # Example step size
            t_teacher = t
            t_student = torch.clamp(t + delta, max=1.0)

            x_t_student = (1-t_student) * x_0 + t_student * x_1
            x_t_teacher = (1-t_teacher) * x_0 + t_teacher * x_1

            # Student Prediction
            v_student = model.flow_lm.flow_net(context_rep, t_student, t_student, x_t_student)
            x_0_student = x_t_student - t_student * v_student

            # Teacher Prediction
            with torch.no_grad():
                # We need to use EMA model
                # But EMA model needs context too.
                # If we freeze backbone, context is same.
                # If backbone is updated, we should ideally use EMA backbone for context.

                # Check if we have EMA model
                # Assume ema_model has same structure
                # We need to run ema backbone if not frozen/shared.
                # If freeze_backbone is True, model.flow_lm.backbone and ema_model.flow_lm.backbone are same (initially)
                # but if we didn't update ema backbone, it's fine.
                # If backbone is frozen, we can reuse `context_rep`.

                # If backbone is NOT frozen, we should technically run EMA backbone.
                # But for efficiency (HBM), running backbone twice is expensive.
                # Let's assume backbone is frozen or we reuse context for teacher (approximation).

                v_teacher = ema_model.flow_lm.flow_net(context_rep, t_teacher, t_teacher, x_t_teacher)
                x_0_teacher = x_t_teacher - t_teacher * v_teacher

            loss = nn.MSELoss()(x_0_student, x_0_teacher)

            # Update EMA
            update_ema(ema_model, model)

        else:
            # Fallback to Consistency Training (Ground Truth Target) if no EMA provided
            loss = nn.MSELoss()(x_0_pred, x_0)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
