from __future__ import annotations

import math

import torch
from torch import Tensor

RESERVOIR_CAPACITY = 10_000
RESERVOIR_SEED = 1234


class SeededReservoir:
    def __init__(self, capacity: int, dim: int, *, seed: int) -> None:
        self.capacity = capacity
        self.buffer = torch.empty(capacity, dim, dtype=torch.float32)
        self.generator = torch.Generator().manual_seed(seed)
        self.n_seen = 0

    def update(self, rows: Tensor) -> None:
        rows = rows.detach().reshape(-1, rows.shape[-1]).to(
            "cpu", torch.float32)
        n = rows.shape[0]
        if n == 0:
            return
        fill = max(0, min(self.capacity - self.n_seen, n))
        if fill:
            self.buffer[self.n_seen:self.n_seen + fill] = rows[:fill]
        if n > fill:
            rest = rows[fill:]
            ranks = torch.arange(self.n_seen + fill, self.n_seen + n,
                                 dtype=torch.float64)
            u = torch.rand(rest.shape[0], generator=self.generator,
                           dtype=torch.float64)
            slots = torch.floor(u * (ranks + 1.0)).to(torch.long)
            entering = (slots < self.capacity).nonzero(as_tuple=True)[0]
            for i in entering.tolist():
                self.buffer[int(slots[i])] = rest[i]
        self.n_seen += n

    def collect(self) -> Tensor:
        if self.n_seen < self.capacity:
            return self.buffer[:self.n_seen]
        return self.buffer


def mean_pairwise_cosine(rows: Tensor) -> float:
    if rows.shape[0] < 2:
        raise ValueError("mean_pairwise_cosine needs at least 2 rows")
    x = rows.to(torch.float64)
    norms = x.norm(dim=1)
    if not bool((norms > 0).all()):
        raise ValueError("mean_pairwise_cosine: zero-norm row in sample")
    u = x / norms[:, None]
    s = u.sum(dim=0)
    n = u.shape[0]
    return float((s.dot(s) - n) / (n * (n - 1)))


def top1_spectral_share(cov: Tensor) -> float:
    ev = torch.linalg.eigvalsh(cov.to(torch.float64)).clamp(min=0.0)
    total = float(ev.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("top1_spectral_share: degenerate spectrum")
    return float(ev[-1] / total)
