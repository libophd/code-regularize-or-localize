from __future__ import annotations

import torch
from torch import Tensor


class RunningChannelMoments:
    def __init__(self, d: int) -> None:
        self.d = d
        self.n = 0
        self.sums = torch.zeros(4, d, dtype=torch.float64)
        self.max_abs = torch.zeros(d, dtype=torch.float64)

    def update(self, x: Tensor) -> None:
        if x.dim() != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"RunningChannelMoments.update expects (N, {self.d}); got {tuple(x.shape)}"
            )
        x64 = x.detach().to("cpu", torch.float64)
        p = x64
        for k in range(4):
            self.sums[k] += p.sum(dim=0)
            if k < 3:
                p = p * x64
        self.max_abs = torch.maximum(self.max_abs, x64.abs().amax(dim=0))
        self.n += x.shape[0]

    def mean(self) -> Tensor:
        return self.sums[0] / self.n

    def std(self) -> Tensor:
        m1 = self.mean()
        var = self.sums[1] / self.n - m1 * m1
        return var.clamp(min=0.0).sqrt()

    def excess_kurtosis(self) -> Tensor:
        n = self.n
        mu = self.mean()
        e2, e3, e4 = self.sums[1] / n, self.sums[2] / n, self.sums[3] / n
        m2 = e2 - mu * mu
        m4 = e4 - 4.0 * mu * e3 + 6.0 * mu * mu * e2 - 3.0 * mu.pow(4)
        degenerate = m2 <= 1e-150
        kurt = m4 / m2.clamp(min=1e-150).pow(2) - 3.0
        return torch.where(degenerate, torch.full_like(kurt, -3.0), kurt)


def scale_dispersion(std: Tensor) -> float:
    med = std.median().clamp(min=1e-300)
    return (std.max() / med).item()


def token_norm_concentration(x: Tensor) -> float:
    if x.dim() != 2:
        raise ValueError(f"token_norm_concentration expects (N, d); got {tuple(x.shape)}")
    norms = x.detach().to(torch.float64).norm(dim=1)
    return (norms.max() / norms.median().clamp(min=1e-300)).item()


def axis_alignment(cov: Tensor) -> float:
    if cov.dim() != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"axis_alignment expects (d, d); got {tuple(cov.shape)}")
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    v1 = eigvecs[:, -1]
    return v1.abs().max().item()


class RunningCovariance:
    def __init__(self, d: int) -> None:
        self.d = d
        self.n = 0
        self.sum1 = torch.zeros(d, dtype=torch.float64)
        self.sum2 = torch.zeros(d, d, dtype=torch.float64)

    def update(self, x: Tensor) -> None:
        if x.dim() != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"RunningCovariance.update expects (N, {self.d}); got {tuple(x.shape)}"
            )
        x64 = x.detach().to("cpu", torch.float64)
        self.sum1 += x64.sum(dim=0)
        self.sum2 += x64.T @ x64
        self.n += x.shape[0]

    def covariance(self) -> Tensor:
        mu = self.sum1 / self.n
        return self.sum2 / self.n - torch.outer(mu, mu)
