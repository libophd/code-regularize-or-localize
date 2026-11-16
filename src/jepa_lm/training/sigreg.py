from __future__ import annotations

import torch
from torch import Tensor

_KNOTS: int = 17
_T_MAX: float = 3.0
_LAMBDA: float = 1.0
_HALF_PRECISION: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)


def _compute_dtype(in_dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if in_dtype in _HALF_PRECISION else in_dtype


def _quadrature(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    t = torch.linspace(0.0, _T_MAX, _KNOTS, device=device, dtype=dtype)
    dt = _T_MAX / (_KNOTS - 1)
    trap = torch.full((_KNOTS,), 2.0 * dt, device=device, dtype=dtype)
    trap[0] = dt
    trap[-1] = dt
    window = torch.exp(-t.square() / (2.0 * _LAMBDA * _LAMBDA))
    phi = torch.exp(-t.square() / 2.0)
    weights = trap * window
    return t, weights, phi


def sample_unit_directions(
    d: int,
    m: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    if d < 1 or m < 1:
        raise ValueError(f"sample_unit_directions requires d >= 1 and m >= 1; got d={d}, m={m}")
    a = torch.randn(d, m, device=device, dtype=dtype, generator=generator)
    a = a / a.norm(p=2, dim=0, keepdim=True)
    return a


def epps_pulley_statistic(
    univariate: Tensor,
    *,
    t: Tensor,
    weights: Tensor,
    phi: Tensor,
) -> Tensor:
    n = univariate.shape[-1]
    x_t = univariate.unsqueeze(-1) * t
    err = (x_t.cos().mean(-2) - phi).square() + x_t.sin().mean(-2).square()
    return (err @ weights) * n


def _resolve_directions(
    d: int,
    m: int,
    directions: Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    if directions is None:
        return sample_unit_directions(d, m, device=device, dtype=dtype, generator=generator)
    if directions.shape != (d, m):
        raise ValueError(
            f"directions shape mismatch: expected ({d}, {m}), got {tuple(directions.shape)}"
        )
    return directions.to(device=device, dtype=dtype)


def sigreg(
    z: Tensor,
    *,
    m: int = 1024,
    directions: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    if z.dim() != 2:
        raise ValueError(f"sigreg expects a 2D tensor (N, D); got shape {tuple(z.shape)}")
    if m < 1:
        raise ValueError(f"m must be >= 1; got {m}")
    n, d = z.shape
    if n < 1 or d < 1:
        raise ValueError(f"sigreg requires N >= 1 and D >= 1; got shape {tuple(z.shape)}")

    compute_dtype = _compute_dtype(z.dtype)
    z = z.to(compute_dtype)
    t, weights, phi = _quadrature(z.device, compute_dtype)
    dirs = _resolve_directions(d, m, directions, device=z.device, dtype=compute_dtype, generator=generator)

    h = (z @ dirs).t()
    return epps_pulley_statistic(h, t=t, weights=weights, phi=phi).mean()


def sigreg_per_position(
    z: Tensor,
    *,
    m: int = 1024,
    directions: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    if z.dim() != 3:
        raise ValueError(f"sigreg_per_position expects a 3D tensor (B, T, D); got shape {tuple(z.shape)}")
    if m < 1:
        raise ValueError(f"m must be >= 1; got {m}")
    b, length, d = z.shape
    if b < 1 or length < 1 or d < 1:
        raise ValueError(f"sigreg_per_position requires B, T, D >= 1; got shape {tuple(z.shape)}")

    compute_dtype = _compute_dtype(z.dtype)
    z = z.to(compute_dtype)
    t, weights, phi = _quadrature(z.device, compute_dtype)
    dirs = _resolve_directions(d, m, directions, device=z.device, dtype=compute_dtype, generator=generator)

    h = z @ dirs
    h = h.permute(2, 1, 0).contiguous()
    return epps_pulley_statistic(h, t=t, weights=weights, phi=phi).mean()
