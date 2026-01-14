import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from typing import NamedTuple

from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule


class KVCacheResult(NamedTuple):
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> "KVCacheResult":
        B, H, T, D = keys.shape
        assert tuple(values.shape[:-1]) == (B, H, T)
        positions = torch.arange(T, device=keys.device, dtype=torch.long)
        return KVCacheResult(keys, values, positions.expand(B, -1))


def complete(
    cache: torch.Tensor, end_offset: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> KVCacheResult:
    capacity = cache.shape[3]
    assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
    B, H, T, D = k.shape
    assert T > 0
    indexes = torch.arange(T, device=end_offset.device, dtype=end_offset.dtype)
    indexes = indexes + end_offset.view(-1, 1)
    indexes = indexes % capacity
    # indexes is [B, T]
    # k is [B, H, T, D]
    # cache is [B, H, T', D]
    this_indexes = indexes.view(B, 1, T, 1)
    this_indexes = this_indexes.expand(-1, H, T, D)
    cache[0].scatter_(2, this_indexes, k)
    cache[1].scatter_(2, this_indexes, v)

    keys = cache[0]
    values = cache[1]

    indexes = torch.arange(capacity, device=end_offset.device, dtype=torch.long)

    # end_index correspond to the actual index where the last value was written.
    last_offset = end_offset.view(-1, 1) + T - 1
    end_index = last_offset % capacity
    delta = indexes - end_index

    positions = torch.where(delta <= 0, last_offset + delta, last_offset + delta - capacity)
    end_offset[:] = end_offset + T
    invalid = indexes >= end_offset.view(-1, 1)
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)

    return KVCacheResult(keys, values, positions)


def complete_kv(
    cache: torch.Tensor, current_end: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    current_end = current_end.shape[0]

    cache[0, :, current_end : current_end + k.shape[1]] = k
    cache[1, :, current_end : current_end + v.shape[1]] = v
    valid = cache[:, :, : current_end + k.shape[1]]
    return valid[0], valid[1]


def _materialize_causal_mask(
    shape: tuple[int, ...], shift: int, device: str | torch.device = "cpu"
) -> torch.Tensor:
    dtype = torch.float32

    num_queries, num_keys = shape[-2:]
    shift = num_keys - num_queries

    tensor = torch.full(shape, dtype=dtype, fill_value=1, device=device)
    mask = torch.tril(tensor, diagonal=shift).to(dtype)
    mask = torch.log(mask)
    return mask.to(dtype)


class StreamingMultiheadAttention(StatefulModule):
    """Similar to `nn.MultiheadAttention` but with support for streaming.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        context (int, optional): Number of time steps the attention can access to.
            Can access `context` time steps into the past.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, rope: RotaryEmbedding, context: int | None = None
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.rope = rope
        self.num_heads = num_heads
        self.context = context

        out_dim = 3 * embed_dim
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        dim_per_head = self.embed_dim // self.num_heads
        state = {}
        state["offset"] = torch.zeros(batch_size, dtype=torch.long)
        state["cache"] = torch.zeros(
            (2, batch_size, self.num_heads, sequence_length, dim_per_head)
        )
        state["end_offset"] = torch.zeros(batch_size, dtype=torch.long)
        return state

    def increment_step(self, state, increment: int = 1):
        state["offset"] += increment

    def _complete_kv(self, k, v, model_state: dict | None):
        if model_state is None:
            # Not streaming
            return KVCacheResult.from_kv(k, v)
        else:
            # Streaming
            layer_state = self.get_state(model_state)
            return complete(layer_state["cache"], layer_state["end_offset"], k, v)

    def _streaming_offset(self, model_state: dict | None, query: torch.Tensor) -> torch.Tensor:
        if model_state is None:
            return torch.zeros(query.shape[0], device=query.device, dtype=torch.long)
        else:
            return self.get_state(model_state)["offset"]

    def forward(self, query: torch.Tensor, model_state: dict | None) -> torch.Tensor:
        B, T = query.shape[:2]
        offset = self._streaming_offset(model_state, query)
        projected = self.in_proj(query)

        # Reshape from (b, t, p*h*d) to (p, b, h, t, d)
        d = self.embed_dim // self.num_heads
        q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads, d=d)

        # Permute from (b, h, t, d) to (b, t, h, d) for rope
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        q, k = self.rope(q, k, offset)
        # Permute back from (b, t, h, d) to (b, h, t, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        k, v, pos_k = self._complete_kv(k, v, model_state)
        pos_k = pos_k[:, None]
        pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=q.device, dtype=torch.long).view(
            -1, 1
        )
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0)
        if self.context is not None:
            attn_bias = attn_bias & (delta < self.context)
        attn_bias = attn_bias[:, None]

        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_proj(x)
        return x
