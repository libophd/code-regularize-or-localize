from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def anisotropy(
    hidden: Tensor,
    n_pairs: int = 10000,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    z = hidden.reshape(-1, hidden.shape[-1]).float()
    n = z.shape[0]
    if n < 2:
        raise ValueError(f"anisotropy: need at least 2 vectors; got {n}")
    if n_pairs < 1:
        raise ValueError(f"anisotropy: n_pairs must be >= 1; got {n_pairs}")

    z_norm = F.normalize(z, dim=-1)

    idx_i = torch.randint(0, n, (n_pairs,), generator=generator, device=z.device)
    idx_j = torch.randint(0, n, (n_pairs,), generator=generator, device=z.device)
    same = idx_i == idx_j
    idx_j = torch.where(same, (idx_j + 1) % n, idx_j)

    cos_sim = (z_norm[idx_i] * z_norm[idx_j]).sum(dim=-1)
    return cos_sim.mean()


def isotropy_spectrum(hidden: Tensor, *, top_k: int | None = None) -> Tensor:
    z = hidden.reshape(-1, hidden.shape[-1]).float()
    if z.shape[0] < 2:
        raise ValueError(f"isotropy_spectrum: need at least 2 vectors; got {z.shape[0]}")
    z_centered = z - z.mean(dim=0, keepdim=True)
    svals = torch.linalg.svdvals(z_centered)
    if top_k is not None:
        svals = svals[:top_k]
    return svals


def calibrated_nll(logits: Tensor, targets: Tensor) -> Tensor:
    raise NotImplementedError


def per_layer_residual_norm(k_or_v: Tensor, predicted: Tensor) -> Tensor:
    if k_or_v.shape != predicted.shape:
        raise ValueError(
            f"per_layer_residual_norm: shape mismatch "
            f"K{tuple(k_or_v.shape)} vs predicted{tuple(predicted.shape)}"
        )
    k = k_or_v.reshape(-1, k_or_v.shape[-1]).float()
    p = predicted.reshape(-1, predicted.shape[-1]).float()
    ref = torch.linalg.norm(k)
    if ref == 0:
        raise ValueError("per_layer_residual_norm: reference K/V has zero norm")
    return torch.linalg.norm(k - p) / ref


def attention_score_frobenius_error(
    q: Tensor,
    k_true: Tensor,
    k_compressed: Tensor,
    *,
    causal: bool = True,
) -> Tensor:
    if not (q.shape[-1] == k_true.shape[-1] == k_compressed.shape[-1]):
        raise ValueError(
            "attention_score_frobenius_error: head-dim mismatch "
            f"q{tuple(q.shape)} k_true{tuple(k_true.shape)} "
            f"k_compressed{tuple(k_compressed.shape)}"
        )
    if k_true.shape != k_compressed.shape:
        raise ValueError(
            "attention_score_frobenius_error: "
            f"k_true{tuple(k_true.shape)} != k_compressed{tuple(k_compressed.shape)}"
        )
    q = q.float()
    k_true = k_true.float()
    k_compressed = k_compressed.float()
    scale = float(q.shape[-1]) ** -0.5

    def _attn(k: Tensor) -> Tensor:
        scores = (q @ k.transpose(-1, -2)) * scale
        if causal:
            t_q, t_k = scores.shape[-2], scores.shape[-1]
            keep = torch.ones(t_q, t_k, dtype=torch.bool, device=scores.device).tril()
            scores = scores.masked_fill(~keep, float("-inf"))
        return torch.softmax(scores, dim=-1)

    a_true = _attn(k_true)
    a_comp = _attn(k_compressed)
    ref = torch.linalg.norm(a_true)
    if ref == 0:
        raise ValueError("attention_score_frobenius_error: true attention map has zero norm")
    return torch.linalg.norm(a_true - a_comp) / ref


def ann_recall_at_k(retrieved: Tensor, ground_truth: Tensor, k: int) -> Tensor:
    raise NotImplementedError


def stp_perpendicular_norm(h: Tensor, triplets: Tensor) -> Tensor:
    raise NotImplementedError


def stp_triplet_cosine(h: Tensor, triplets: Tensor) -> Tensor:
    raise NotImplementedError


def henaff_temporal_straightening(z: Tensor) -> Tensor:
    raise NotImplementedError
