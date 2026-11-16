from __future__ import annotations

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, positions: Tensor, *, base: float = 10000.0) -> Tensor:
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"apply_rope: head_dim must be even, got {head_dim}")
    if positions.shape[-1] != x.shape[-2]:
        raise ValueError(
            f"apply_rope: positions length {tuple(positions.shape)} != "
            f"sequence length T={x.shape[-2]}"
        )
    device = x.device
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    angles = positions.to(device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((angles, angles), dim=-1)
    cos = emb.cos().to(x.dtype)
    sin = emb.sin().to(x.dtype)
    return x * cos + rotate_half(x) * sin
