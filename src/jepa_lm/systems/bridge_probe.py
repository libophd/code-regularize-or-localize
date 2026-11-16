from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW

from jepa_lm.eval.metrics import attention_score_frobenius_error, per_layer_residual_norm
from jepa_lm.systems.kv_bridge import (
    KV_BRIDGES,
    KVBridge,
    PerLayerKVPredictorBridge,
    PostHocKVDecoderBridge,
)
from jepa_lm.systems.kv_capture import KVCapture
from jepa_lm.systems.rope import apply_rope


@dataclass
class BridgeProbeResult:
    probe: str
    n_steps_trained: int
    gate_metric: str
    gate_value: float
    gate_threshold: float
    passed: bool
    per_layer: dict[str, list[float]] = field(default_factory=dict)
    last_train_loss: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "n_steps_trained": self.n_steps_trained,
            "gate_metric": self.gate_metric,
            "gate_value": self.gate_value,
            "gate_threshold": self.gate_threshold,
            "passed": self.passed,
            "per_layer": self.per_layer,
            "last_train_loss": self.last_train_loss,
            "extra": self.extra,
        }


class FrozenTrunkProbe:
    def __init__(
        self,
        trunk: nn.Module,
        bridge: KVBridge,
        *,
        device: torch.device | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        grad_clip: float = 1.0,
        needs_z: bool = False,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.trunk = trunk.to(self.device)
        self.bridge = bridge.to(self.device)
        self.grad_clip = grad_clip
        self.needs_z = needs_z

        self.trunk.eval()
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        bridge_params = [p for p in self.bridge.parameters() if p.requires_grad]
        if not bridge_params:
            raise ValueError(
                f"{type(bridge).__name__} has no trainable parameters; a frozen-trunk "
                "probe needs a bridge that owns parameters (predictors / decoders)."
            )
        self.optimizer = AdamW(bridge_params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.global_step = 0

    def _capture(self, input_ids: Tensor) -> tuple[list[Tensor], list[Tensor], Tensor]:
        with KVCapture(self.trunk) as cap, torch.no_grad():
            h = self.trunk(input_ids)
            k, v = cap.captured()
        return list(k), list(v), h

    def train_step(self, raw_batch: Mapping[str, Tensor]) -> float:
        input_ids = raw_batch["input_ids"].to(self.device)
        k, v, h = self._capture(input_ids)

        self.bridge.train()
        z = h if self.needs_z else None
        loss = self.bridge.auxiliary_loss(k, v, z=z)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.grad_clip)
        self.optimizer.step()
        self.global_step += 1
        return loss.detach().item()

    def train(
        self,
        dataloader: Iterable[Mapping[str, Tensor]],
        *,
        n_steps: int,
        on_step: Callable[[int, float], None] | None = None,
    ) -> float:
        last = 0.0
        for raw_batch in dataloader:
            if self.global_step >= n_steps:
                break
            last = self.train_step(raw_batch)
            if on_step is not None:
                on_step(self.global_step, last)
        return last


def gate_per_layer_residual_norm(
    trunk: nn.Module,
    bridge: PerLayerKVPredictorBridge,
    dataloader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device | None = None,
) -> tuple[float, dict[str, list[float]]]:
    device = device or next(trunk.parameters()).device
    n_layers = bridge.n_layers
    k_sum = [0.0] * n_layers
    v_sum = [0.0] * n_layers
    weight = 0.0

    bridge.eval()
    with torch.no_grad():
        for raw_batch in dataloader:
            input_ids = raw_batch["input_ids"].to(device)
            with KVCapture(trunk) as cap:
                trunk(input_ids)
                k, v = cap.captured()
            if k[0].shape[1] < 2:
                raise ValueError(
                    "gate_per_layer_residual_norm needs T>=2 for next-position prediction; "
                    f"got T={k[0].shape[1]}"
                )
            n_pred = k[0].shape[0] * (k[0].shape[1] - 1)
            for layer in range(n_layers):
                k_res = per_layer_residual_norm(
                    k[layer][:, 1:, :], bridge.k_predictors[layer](k[layer][:, :-1, :])
                ).item()
                v_res = per_layer_residual_norm(
                    v[layer][:, 1:, :], bridge.v_predictors[layer](v[layer][:, :-1, :])
                ).item()
                k_sum[layer] += k_res * n_pred
                v_sum[layer] += v_res * n_pred
            weight += n_pred

    if weight == 0.0:
        raise ValueError("gate_per_layer_residual_norm: empty calibration loader")
    k_per_layer = [s / weight for s in k_sum]
    v_per_layer = [s / weight for s in v_sum]
    mean = float(sum(k_per_layer + v_per_layer) / (2 * n_layers))
    return mean, {"k_residual": k_per_layer, "v_residual": v_per_layer}


def run_a2_probe(
    trunk: nn.Module,
    *,
    n_steps: int,
    train_loader: Iterable[Mapping[str, Tensor]],
    eval_loader: Iterable[Mapping[str, Tensor]],
    device: torch.device | None = None,
    lr: float = 1e-3,
    bridge_name: str = "per_layer_kv_predictors",
    bridge_kwargs: Mapping[str, Any] | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> BridgeProbeResult:
    cap = KVCapture(trunk)
    bridge = KV_BRIDGES.build(
        bridge_name, n_layers=cap.n_layers, d_kv=cap.d_kv, **dict(bridge_kwargs or {})
    )
    if not isinstance(bridge, PerLayerKVPredictorBridge):
        raise ValueError(
            f"run_a2_probe: bridge {bridge_name!r} is {type(bridge).__name__}, not a "
            f"PerLayerKVPredictorBridge — the A2' gate walks k_predictors/v_predictors."
        )
    probe = FrozenTrunkProbe(trunk, bridge, device=device, lr=lr, needs_z=False)
    last_loss = probe.train(train_loader, n_steps=n_steps, on_step=on_step)
    gate_value, per_layer = gate_per_layer_residual_norm(
        trunk, bridge, eval_loader, device=probe.device
    )
    threshold = 0.5
    return BridgeProbeResult(
        probe=f"a2_{bridge_name}",
        n_steps_trained=probe.global_step,
        gate_metric="mean_per_layer_residual_norm",
        gate_value=gate_value,
        gate_threshold=threshold,
        passed=gate_value < threshold,
        per_layer=per_layer,
        last_train_loss=last_loss,
    )


def _split_heads(x: Tensor, n_heads: int) -> Tensor:
    b, t, d = x.shape
    head_dim = d // n_heads
    return x.view(b, t, n_heads, head_dim).permute(0, 2, 1, 3)


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


def gate_attention_score_frobenius_error(
    trunk: nn.Module,
    bridge: PostHocKVDecoderBridge,
    dataloader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device | None = None,
) -> tuple[float, dict[str, list[float]]]:
    device = device or next(trunk.parameters()).device
    cfg = trunk.model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    base = float(getattr(cfg, "rope_theta", 10000.0))
    rope_scaling = getattr(cfg, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type") or "default"
    else:
        rope_type = "default" if rope_scaling is None else str(rope_scaling)
    partial = getattr(cfg, "partial_rotary_factor", 1.0)
    if rope_type != "default" or partial != 1.0:
        raise NotImplementedError(
            "gate_attention_score_frobenius_error: standalone RoPE assumes full, "
            f"unscaled rotary, but config has rope_type={rope_type!r}, "
            f"partial_rotary_factor={partial}. Extend systems.rope before running "
            "the A3' gate on this checkpoint."
        )
    if n_heads % n_kv != 0:
        raise ValueError(
            f"gate: num_attention_heads {n_heads} not a multiple of "
            f"num_key_value_heads {n_kv}"
        )
    n_rep = n_heads // n_kv
    n_layers = bridge.n_layers

    sums = [0.0] * n_layers
    weight = 0.0
    bridge.eval()
    with torch.no_grad():
        for raw_batch in dataloader:
            input_ids = raw_batch["input_ids"].to(device)
            with KVCapture(trunk, capture_q=True) as cap:
                h = trunk(input_ids)
                k, _v = cap.captured()
                q = cap.captured_q()
            b, t, _ = h.shape
            positions = torch.arange(t, device=device)
            for layer in range(n_layers):
                qh = apply_rope(_split_heads(q[layer], n_heads), positions, base=base)
                kth = _repeat_kv(apply_rope(_split_heads(k[layer], n_kv), positions, base=base), n_rep)
                k_dec = bridge.k_decoders[layer](h)
                kdh = _repeat_kv(apply_rope(_split_heads(k_dec, n_kv), positions, base=base), n_rep)
                err = attention_score_frobenius_error(qh, kth, kdh, causal=True).item()
                sums[layer] += err * b
            weight += b

    if weight == 0.0:
        raise ValueError("gate_attention_score_frobenius_error: empty calibration loader")
    per_layer = [s / weight for s in sums]
    mean = float(sum(per_layer) / n_layers)
    return mean, {"attn_frobenius": per_layer}


def run_a3_probe(
    trunk: nn.Module,
    *,
    n_steps: int,
    train_loader: Iterable[Mapping[str, Tensor]],
    eval_loader: Iterable[Mapping[str, Tensor]],
    device: torch.device | None = None,
    lr: float = 1e-3,
    on_step: Callable[[int, float], None] | None = None,
) -> BridgeProbeResult:
    cap = KVCapture(trunk)
    d_proj = trunk.model.config.hidden_size
    bridge = PostHocKVDecoderBridge(n_layers=cap.n_layers, d_kv=cap.d_kv, d_proj=d_proj)
    probe = FrozenTrunkProbe(trunk, bridge, device=device, lr=lr, needs_z=True)
    last_loss = probe.train(train_loader, n_steps=n_steps, on_step=on_step)
    gate_value, per_layer = gate_attention_score_frobenius_error(
        trunk, bridge, eval_loader, device=probe.device
    )
    threshold = 0.1
    return BridgeProbeResult(
        probe="a3_post_hoc_kv_decoders",
        n_steps_trained=probe.global_step,
        gate_metric="mean_attention_score_frobenius_error",
        gate_value=gate_value,
        gate_threshold=threshold,
        passed=gate_value <= threshold,
        per_layer=per_layer,
        last_train_loss=last_loss,
    )
