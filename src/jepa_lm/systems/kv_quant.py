from __future__ import annotations

from contextlib import AbstractContextManager
from functools import lru_cache

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from jepa_lm.systems.kv_capture import attention_projections

GRANULARITIES = ("per_token", "per_channel")
TARGETS = ("k", "v", "both")
MODES = ("symmetric", "asymmetric")

_ROTATION_SEED = 0


@lru_cache(maxsize=8)
def _rotation_matrix_cpu(d: int) -> Tensor:
    gen = torch.Generator().manual_seed(_ROTATION_SEED)
    gaussian = torch.randn(d, d, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    return q * r.diagonal().sign()


def rotation_matrix(d: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return _rotation_matrix_cpu(d).to(device=device, dtype=dtype)


def _group_dims(x: Tensor, granularity: str) -> tuple[int, ...]:
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {GRANULARITIES}, got {granularity!r}"
        )
    return (-1,) if granularity == "per_token" else tuple(range(x.dim() - 1))


def _grouped_view(x: Tensor, granularity: str, group_size: int) -> tuple[Tensor, int]:
    if x.dim() != 3:
        raise ValueError(
            f"fake_quantize: group_size requires a (B, T, d) tensor, got {x.dim()}-D"
        )
    b, t, d = x.shape
    if granularity == "per_token":
        if d % group_size:
            raise ValueError(
                f"fake_quantize: channel dim {d} not divisible by group_size {group_size}"
            )
        return x.reshape(b, t, d // group_size, group_size), -1
    if t % group_size:
        raise ValueError(
            f"fake_quantize: position dim {t} not divisible by group_size {group_size}"
        )
    return x.reshape(b, t // group_size, group_size, d), 2


def _quantize_on_grid(x: Tensor, bits: int, dims: tuple[int, ...], mode: str) -> Tensor:
    tiny = torch.finfo(x.dtype).tiny
    if mode == "symmetric":
        qmax = 2 ** (bits - 1) - 1
        absmax = x.abs().amax(dim=dims, keepdim=True)
        scale = (absmax / qmax).clamp(min=tiny)
        return (x / scale).round_().clamp_(-qmax, qmax).mul_(scale)
    n_levels = 2**bits - 1
    lo = x.amin(dim=dims, keepdim=True)
    hi = x.amax(dim=dims, keepdim=True)
    scale = ((hi - lo) / n_levels).clamp(min=tiny)
    return ((x - lo) / scale).round_().clamp_(0, n_levels).mul_(scale).add_(lo)


def fake_quantize(
    x: Tensor,
    bits: int,
    granularity: str,
    mode: str = "symmetric",
    group_size: int | None = None,
) -> Tensor:
    if not 2 <= bits <= 8:
        raise ValueError(f"fake_quantize: bits must be in [2, 8], got {bits}")
    if mode not in MODES:
        raise ValueError(f"fake_quantize: mode must be one of {MODES}, got {mode!r}")
    if group_size is None:
        return _quantize_on_grid(x, bits, _group_dims(x, granularity), mode)
    if group_size < 1:
        raise ValueError(f"fake_quantize: group_size must be >= 1, got {group_size}")
    _group_dims(x, granularity)
    grouped, dim = _grouped_view(x, granularity, group_size)
    return _quantize_on_grid(grouped, bits, (dim,), mode).reshape(x.shape)


class KVQuantize(AbstractContextManager):
    def __init__(
        self,
        trunk: nn.Module,
        *,
        bits: int,
        granularity: str = "per_token",
        targets: str = "both",
        mode: str = "symmetric",
        subtract_mean: bool = False,
        group_size: int | None = None,
        standardize: bool = False,
        rotate: bool = False,
        layers: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        fake_quantize(torch.zeros(1, 1, 2), bits, granularity, mode)
        if targets not in TARGETS:
            raise ValueError(
                f"KVQuantize: targets must be one of {TARGETS}, got {targets!r}"
            )
        if group_size is not None and group_size < 1:
            raise ValueError(
                f"KVQuantize: group_size must be >= 1, got {group_size}"
            )
        if subtract_mean and standardize:
            raise ValueError(
                "KVQuantize: standardize subsumes subtract_mean; pass only one"
            )
        self._pairs = attention_projections(trunk)
        if layers is not None:
            n = len(self._pairs)
            if not layers:
                raise ValueError("KVQuantize: layers must be non-empty "
                                 "when given")
            for ell in layers:
                if type(ell) is not int:
                    raise ValueError(
                        f"KVQuantize: layers entries must be literal ints, "
                        f"got {ell!r}")
                if not 0 <= ell < n:
                    raise ValueError(
                        f"KVQuantize: layer {ell} out of range [0, {n})")
            if len(set(layers)) != len(layers):
                raise ValueError("KVQuantize: duplicate layer indices")
        self.layers = tuple(layers) if layers is not None else None
        self.bits = bits
        self.granularity = granularity
        self.targets = targets
        self.mode = mode
        self.subtract_mean = subtract_mean
        self.group_size = group_size
        self.standardize = standardize
        self.rotate = rotate
        self._handles: list[RemovableHandle] = []

    def _hook(self, _module: nn.Module, _inputs: tuple, output: Tensor) -> Tensor:
        x = output
        mean = std = q = None
        if self.subtract_mean or self.standardize:
            dims = tuple(range(x.dim() - 1))
            mean = x.mean(dim=dims, keepdim=True)
            x = x - mean
            if self.standardize:
                std = x.std(dim=dims, keepdim=True).clamp(
                    min=torch.finfo(x.dtype).tiny
                )
                x = x / std
        if self.rotate:
            q = rotation_matrix(x.shape[-1], x.device, x.dtype)
            x = x @ q
        y = fake_quantize(
            x, self.bits, self.granularity, self.mode, group_size=self.group_size
        )
        if q is not None:
            y = y @ q.T
        if std is not None:
            y = y * std
        if mean is not None:
            y = y + mean
        return y

    def __enter__(self) -> KVQuantize:
        for idx, (k_proj, v_proj) in enumerate(self._pairs):
            if self.layers is not None and idx not in self.layers:
                continue
            if self.targets in ("k", "both"):
                self._handles.append(k_proj.register_forward_hook(self._hook))
            if self.targets in ("v", "both"):
                self._handles.append(v_proj.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class PostRopeKQuantize(AbstractContextManager):
    def __init__(
        self,
        *,
        bits: int,
        granularity: str = "per_token",
        mode: str = "symmetric",
        group_size: int | None = None,
    ) -> None:
        fake_quantize(torch.zeros(1, 1, 2), bits, granularity, mode)
        if group_size is not None and group_size < 1:
            raise ValueError(
                f"PostRopeKQuantize: group_size must be >= 1, got {group_size}"
            )
        self.bits = bits
        self.granularity = granularity
        self.mode = mode
        self.group_size = group_size
        self._module = None
        self._original = None

    def __enter__(self) -> PostRopeKQuantize:
        from transformers.models.llama import modeling_llama

        if getattr(modeling_llama.apply_rotary_pos_emb, "_kv_quant_patch", False):
            raise RuntimeError("PostRopeKQuantize: already active in this process")
        self._module = modeling_llama
        self._original = modeling_llama.apply_rotary_pos_emb
        original = self._original

        def patched(q, k, cos, sin, unsqueeze_dim=1):
            q_embed, k_embed = original(q, k, cos, sin, unsqueeze_dim)
            b, h, t, dh = k_embed.shape
            flat = k_embed.transpose(1, 2).reshape(b, t, h * dh)
            quant = fake_quantize(
                flat, self.bits, self.granularity, self.mode,
                group_size=self.group_size,
            )
            return q_embed, quant.reshape(b, t, h, dh).transpose(1, 2)

        patched._kv_quant_patch = True
        modeling_llama.apply_rotary_pos_emb = patched
        return self

    def __exit__(self, *exc: object) -> None:
        if self._module is not None:
            self._module.apply_rotary_pos_emb = self._original
            self._module = None
            self._original = None
