from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from itertools import chain

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW

from jepa_lm.eval.evaluator import evaluate
from jepa_lm.eval.metrics import anisotropy, isotropy_spectrum
from jepa_lm.systems.bridge_probe import BridgeProbeResult
from jepa_lm.systems.kv_bridge import MultiLayerSIGRegBridge
from jepa_lm.systems.kv_capture import KVCapture, attention_projections
from jepa_lm.systems.lora import LORA_KV_SPEC, LoraSpec, inject_lora

_UNFREEZE_MODES = ("trunk", "kv_proj", "lora")


class A1Probe:
    def __init__(
        self,
        trunk: nn.Module,
        head: nn.Module,
        base_loss_module: nn.Module,
        *,
        lambda_kv: float = 0.01,
        m_projections: int = 1024,
        device: torch.device | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        grad_clip: float = 1.0,
        unfreeze: str = "trunk",
        amp_dtype: torch.dtype | None = None,
        lora_spec: LoraSpec | None = None,
    ) -> None:
        if unfreeze not in _UNFREEZE_MODES:
            raise ValueError(f"unfreeze must be one of {_UNFREEZE_MODES}; got {unfreeze!r}")
        if lora_spec is not None and unfreeze != "lora":
            raise ValueError("lora_spec is only meaningful with unfreeze='lora'")
        self.lora_spec = lora_spec if lora_spec is not None else LORA_KV_SPEC
        self.device = device or torch.device("cpu")
        self.trunk = trunk.to(self.device)
        self.head = head.to(self.device)
        self.base_loss = base_loss_module.to(self.device)
        self.grad_clip = grad_clip
        self.amp_dtype = amp_dtype

        cap = KVCapture(trunk)
        self.bridge = MultiLayerSIGRegBridge(
            n_layers=cap.n_layers, d_kv=cap.d_kv, lambda_kv=lambda_kv, m_projections=m_projections
        ).to(self.device)

        trainable = self._select_trainable(unfreeze)
        if not trainable:
            raise ValueError(f"A1Probe(unfreeze={unfreeze!r}) selected no trainable parameters")
        self.optimizer = AdamW(trainable, lr=lr, betas=betas, weight_decay=weight_decay)
        self.global_step = 0

    def _select_trainable(self, unfreeze: str) -> list[nn.Parameter]:
        if unfreeze == "trunk":
            for p in chain(self.trunk.parameters(), self.head.parameters()):
                p.requires_grad_(True)
            return list(
                {id(p): p for p in chain(self.trunk.parameters(), self.head.parameters())}.values()
            )
        for p in chain(self.trunk.parameters(), self.head.parameters()):
            p.requires_grad_(False)
        if unfreeze == "lora":
            return inject_lora(self.trunk, self.lora_spec)
        trainable: list[nn.Parameter] = []
        for k_proj, v_proj in attention_projections(self.trunk):
            for p in chain(k_proj.parameters(), v_proj.parameters()):
                p.requires_grad_(True)
                trainable.append(p)
        return trainable

    def step(self, raw_batch: Mapping[str, Tensor]) -> dict[str, float]:
        self.trunk.train()
        self.head.train()
        input_ids = raw_batch["input_ids"].to(self.device)
        labels = raw_batch.get("labels", input_ids).to(self.device)

        ctx = (
            torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype)
            if self.amp_dtype is not None
            else nullcontext()
        )
        with ctx:
            with KVCapture(self.trunk) as cap:
                h = self.trunk(input_ids)
                logits = self.head(h)
                k, v = cap.captured()
            base, _components = self.base_loss(
                {"input_ids": input_ids, "labels": labels, "h": h, "logits": logits}
            )
            kv = self.bridge.auxiliary_loss(list(k), list(v))
            total = base + kv

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.grad_clip)
        self.optimizer.step()
        self.global_step += 1
        return {
            "total": total.detach().item(),
            "base": base.detach().item(),
            "kv": kv.detach().item(),
        }

    def train(
        self,
        dataloader: Iterable[Mapping[str, Tensor]],
        *,
        n_steps: int,
        on_step: Callable[[int, dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        last: dict[str, float] = {}
        for raw_batch in dataloader:
            if self.global_step >= n_steps:
                break
            last = self.step(raw_batch)
            if on_step is not None:
                on_step(self.global_step, last)
        return last


def measure_per_layer_kv_geometry(
    trunk: nn.Module,
    dataloader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device | None = None,
    max_vectors: int = 8192,
    n_pairs: int = 20000,
    seed: int = 1234,
) -> dict[str, list[float]]:
    device = device or next(trunk.parameters()).device
    n_layers = KVCapture(trunk).n_layers
    k_bufs: list[list[Tensor]] = [[] for _ in range(n_layers)]
    v_bufs: list[list[Tensor]] = [[] for _ in range(n_layers)]
    counts = [0] * n_layers

    trunk.eval()
    with torch.no_grad():
        for raw_batch in dataloader:
            if all(c >= max_vectors for c in counts):
                break
            input_ids = raw_batch["input_ids"].to(device)
            with KVCapture(trunk) as cap:
                trunk(input_ids)
                k, v = cap.captured()
            for layer in range(n_layers):
                if counts[layer] >= max_vectors:
                    continue
                k_bufs[layer].append(k[layer].reshape(-1, k[layer].shape[-1]).float().cpu())
                v_bufs[layer].append(v[layer].reshape(-1, v[layer].shape[-1]).float().cpu())
                counts[layer] += k_bufs[layer][-1].shape[0]

    if any(c < 2 for c in counts):
        raise ValueError("measure_per_layer_kv_geometry: <2 vectors collected for some layer")

    def _top1(z: Tensor) -> float:
        s_sq = isotropy_spectrum(z).square()
        total = s_sq.sum().item()
        return (s_sq[0] / s_sq.sum()).item() if total > 0.0 else 1.0

    gen = torch.Generator().manual_seed(seed)
    k_aniso, v_aniso, k_top1, v_top1 = [], [], [], []
    for layer in range(n_layers):
        zk = torch.cat(k_bufs[layer])[:max_vectors]
        zv = torch.cat(v_bufs[layer])[:max_vectors]
        k_aniso.append(anisotropy(zk, n_pairs=n_pairs, generator=gen).item())
        v_aniso.append(anisotropy(zv, n_pairs=n_pairs, generator=gen).item())
        k_top1.append(_top1(zk))
        v_top1.append(_top1(zv))
    return {"k_aniso": k_aniso, "v_aniso": v_aniso, "k_top1": k_top1, "v_top1": v_top1}


def run_a1_probe(
    trunk: nn.Module,
    head: nn.Module,
    base_loss_module: nn.Module,
    *,
    n_steps: int,
    train_loader: Iterable[Mapping[str, Tensor]],
    eval_loader_factory: Callable[[], Iterable[Mapping[str, Tensor]]],
    lambda_kv: float = 0.01,
    m_projections: int = 1024,
    device: torch.device | None = None,
    lr: float = 1e-4,
    unfreeze: str = "trunk",
    amp_dtype: torch.dtype | None = None,
    on_step: Callable[[int, dict[str, float]], None] | None = None,
    lora_spec: LoraSpec | None = None,
    on_trained: Callable[[A1Probe], None] | None = None,
    expected_steps: int | None = None,
    aniso_reduction_threshold: float = 0.30,
    nll_delta_threshold: float = 0.15,
    aniso_floor: float = 0.01,
    aniso_seed: int = 1234,
) -> BridgeProbeResult:
    dev = device or next(trunk.parameters()).device

    geo0 = measure_per_layer_kv_geometry(trunk, eval_loader_factory(), device=dev, seed=aniso_seed)
    nll_before = evaluate(trunk, head, eval_loader_factory(), device=dev)["mean_nll"]
    n_meas = 2 * len(geo0["k_aniso"])
    aniso_before = float(sum(geo0["k_aniso"] + geo0["v_aniso"]) / n_meas)

    probe = A1Probe(
        trunk, head, base_loss_module,
        lambda_kv=lambda_kv, m_projections=m_projections,
        device=dev, lr=lr, unfreeze=unfreeze, amp_dtype=amp_dtype,
        lora_spec=lora_spec,
    )
    last = probe.train(train_loader, n_steps=n_steps, on_step=on_step)
    if on_trained is not None:
        on_trained(probe)
    if expected_steps is not None and probe.global_step != expected_steps:
        raise RuntimeError(
            f"run_a1_probe({unfreeze}): trained {probe.global_step} of "
            f"{expected_steps} expected steps (data stream exhausted?) — "
            "refusing to evaluate as a completed run"
        )

    geo1 = measure_per_layer_kv_geometry(trunk, eval_loader_factory(), device=dev, seed=aniso_seed)
    nll_after = evaluate(trunk, head, eval_loader_factory(), device=dev)["mean_nll"]
    aniso_after = float(sum(geo1["k_aniso"] + geo1["v_aniso"]) / n_meas)

    reliable = aniso_before > aniso_floor
    reduction = (aniso_before - aniso_after) / aniso_before if reliable else 0.0
    nll_delta = nll_after - nll_before
    aniso_ok = reliable and reduction >= aniso_reduction_threshold
    nll_ok = nll_delta <= nll_delta_threshold

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs))

    return BridgeProbeResult(
        probe=f"a1_multi_layer_sigreg_{unfreeze}",
        n_steps_trained=probe.global_step,
        gate_metric="per_layer_kv_anisotropy_reduction",
        gate_value=reduction,
        gate_threshold=aniso_reduction_threshold,
        passed=bool(aniso_ok and nll_ok),
        per_layer={
            "k_aniso_before": geo0["k_aniso"], "v_aniso_before": geo0["v_aniso"],
            "k_aniso_after": geo1["k_aniso"], "v_aniso_after": geo1["v_aniso"],
            "k_top1_before": geo0["k_top1"], "v_top1_before": geo0["v_top1"],
            "k_top1_after": geo1["k_top1"], "v_top1_after": geo1["v_top1"],
        },
        last_train_loss=last.get("total"),
        extra={
            "aniso_before": aniso_before,
            "aniso_after": aniso_after,
            "aniso_before_reliable": reliable,
            "aniso_floor": aniso_floor,
            "spectrum_top1_before": _mean(geo0["k_top1"] + geo0["v_top1"]),
            "spectrum_top1_after": _mean(geo1["k_top1"] + geo1["v_top1"]),
            "nll_before": nll_before,
            "nll_after": nll_after,
            "nll_delta": nll_delta,
            "nll_delta_threshold": nll_delta_threshold,
            "anisotropy_ok": aniso_ok,
            "nll_ok": nll_ok,
        },
    )
