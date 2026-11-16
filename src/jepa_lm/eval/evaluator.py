from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from jepa_lm.eval.metrics import anisotropy, isotropy_spectrum
from jepa_lm.token_io.lm_heads import LMHead
from jepa_lm.trunks.llama import Trunk


class _ReservoirSampler:
    def __init__(self, capacity: int, dim: int, *, dtype: torch.dtype, device: torch.device) -> None:
        if capacity < 2:
            raise ValueError(f"_ReservoirSampler capacity must be >= 2; got {capacity}")
        self.capacity = capacity
        self.buffer = torch.empty((capacity, dim), dtype=dtype, device=device)
        self.total_seen = 0

    def update(self, new_vectors: Tensor) -> None:
        n_new = new_vectors.shape[0]
        if n_new == 0:
            return
        capacity = self.capacity
        device = new_vectors.device
        new_positions = self.total_seen + torch.arange(n_new, device=device)

        warmup_mask = new_positions < capacity
        if warmup_mask.any():
            warmup_local = warmup_mask.nonzero(as_tuple=False).reshape(-1)
            warmup_slots = new_positions[warmup_mask]
            self.buffer[warmup_slots] = new_vectors[warmup_local]

        post_mask = ~warmup_mask
        if post_mask.any():
            post_local = post_mask.nonzero(as_tuple=False).reshape(-1)
            post_positions = new_positions[post_mask]
            ranks = (post_positions + 1).float()
            enter_probs = float(capacity) / ranks
            enter = torch.rand(post_local.shape[0], device=device) < enter_probs
            entering = post_local[enter]
            if entering.numel() > 0:
                slots = torch.randint(0, capacity, (entering.shape[0],), device=device)
                self.buffer[slots] = new_vectors[entering]

        self.total_seen += n_new

    def collect(self) -> Tensor:
        return self.buffer[: min(self.total_seen, self.capacity)]


def evaluate(
    trunk: Trunk,
    head: LMHead,
    dataloader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device | None = None,
    ignore_index: int = -100,
    anisotropy_buffer_size: int = 4096,
    anisotropy_n_pairs: int = 10000,
) -> dict[str, float]:
    if anisotropy_buffer_size < 2:
        raise ValueError(f"anisotropy_buffer_size must be >= 2; got {anisotropy_buffer_size}")

    if device is None:
        try:
            device = next(trunk.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    trunk.eval()
    head.eval()

    sum_nll = torch.zeros((), device=device)
    n_tokens = torch.zeros((), device=device)
    sampler: _ReservoirSampler | None = None

    with torch.no_grad():
        for raw_batch in dataloader:
            input_ids = raw_batch["input_ids"].to(device)
            labels = raw_batch.get("labels", input_ids).to(device)
            h = trunk(input_ids)
            logits = head(h)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid = shift_labels != ignore_index
            safe_labels = torch.where(
                valid, shift_labels, torch.full_like(shift_labels, ignore_index)
            )
            per_token = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                safe_labels.reshape(-1),
                ignore_index=ignore_index,
                reduction="none",
            )
            valid_f = valid.reshape(-1).to(per_token.dtype)
            sum_nll = sum_nll + (per_token * valid_f).sum()
            n_tokens = n_tokens + valid_f.sum()

            valid_pos = labels != ignore_index
            valid_h = h[valid_pos]
            if valid_h.shape[0] > 0:
                if sampler is None:
                    sampler = _ReservoirSampler(
                        capacity=anisotropy_buffer_size,
                        dim=valid_h.shape[1],
                        dtype=valid_h.dtype,
                        device=valid_h.device,
                    )
                sampler.update(valid_h)

    if n_tokens.item() == 0:
        raise ValueError("evaluate: no valid tokens in the dataloader")
    if sampler is None or sampler.total_seen < 2:
        raise ValueError(
            f"evaluate: reservoir collected "
            f"{0 if sampler is None else sampler.total_seen} valid hidden states (<2); "
            f"empty, single-token, or fully-ignored dataloader"
        )

    mean_nll = (sum_nll / n_tokens).item()
    perplexity = math.exp(mean_nll)

    h_all = sampler.collect()
    aniso = anisotropy(h_all, n_pairs=anisotropy_n_pairs).item()
    spectrum = isotropy_spectrum(h_all)
    spectrum_sq = spectrum.square()
    total_var = spectrum_sq.sum().item()
    if total_var == 0.0:
        spectrum_top1_share = 1.0
    else:
        spectrum_top1_share = (spectrum_sq[0] / spectrum_sq.sum()).item()

    return {
        "perplexity": perplexity,
        "mean_nll": mean_nll,
        "n_tokens": int(n_tokens.item()),
        "anisotropy_h": aniso,
        "spectrum_top1_share": spectrum_top1_share,
        "spectrum_max": spectrum[0].item(),
        "spectrum_min": spectrum[-1].item(),
        "n_h_collected": h_all.shape[0],
    }
