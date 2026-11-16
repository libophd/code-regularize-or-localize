from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from jepa_lm.eval.evaluator import evaluate
from jepa_lm.systems.a1_probe import A1Probe, measure_per_layer_kv_geometry
from jepa_lm.systems.bridge_probe import BridgeProbeResult
from jepa_lm.systems.kv_bridge import KVBridge, MultiLayerSIGRegBridge
from jepa_lm.training.sigreg import sigreg_per_position

ARMS = ("ce_only", "h_k_only", "h_v_only")
_BRIDGE_TARGETS = ("kv", "k", "v")


class TargetedSIGRegBridge(MultiLayerSIGRegBridge):
    def __init__(
        self,
        n_layers: int,
        d_kv: int,
        lambda_kv: float = 0.01,
        m_projections: int = 1024,
        targets: str = "kv",
    ) -> None:
        super().__init__(n_layers, d_kv, lambda_kv=lambda_kv, m_projections=m_projections)
        if targets not in _BRIDGE_TARGETS:
            raise ValueError(f"targets must be one of {_BRIDGE_TARGETS}; got {targets!r}")
        self.targets = targets

    def auxiliary_loss(
        self,
        k_per_layer: list[Tensor],
        v_per_layer: list[Tensor],
        *,
        z: Tensor | None = None,
    ) -> Tensor:
        if self.targets == "kv":
            return super().auxiliary_loss(k_per_layer, v_per_layer, z=z)
        self._check_layers(k_per_layer, v_per_layer)
        chosen = k_per_layer if self.targets == "k" else v_per_layer
        terms = [sigreg_per_position(t, m=self.m_projections) for t in chosen]
        return self.lambda_kv * torch.stack(terms).sum()


class NullBridge(KVBridge):
    def auxiliary_loss(
        self,
        k_per_layer: list[Tensor],
        v_per_layer: list[Tensor],
        *,
        z: Tensor | None = None,
    ) -> Tensor:
        self._check_layers(k_per_layer, v_per_layer)
        return k_per_layer[0].new_zeros(())


class CEOnlyBaseLoss(nn.Module):
    CE_TERM = "l_aux"

    def __init__(self, checkpoint_loss: nn.Module) -> None:
        super().__init__()
        terms = getattr(checkpoint_loss, "terms", None)
        weights = getattr(checkpoint_loss, "weights", None)
        if terms is None or weights is None or self.CE_TERM not in terms:
            raise ValueError(
                "CEOnlyBaseLoss expects the checkpoint's JEPALMLoss "
                f"(a ModuleDict of terms including {self.CE_TERM!r}); got "
                f"{type(checkpoint_loss).__name__}"
            )
        self.ce_term = terms[self.CE_TERM]
        self.ce_weight = float(weights[self.CE_TERM])
        self.dropped_terms = {
            name: float(w) for name, w in weights.items() if name != self.CE_TERM
        }

    def forward(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        value = self.ce_term(batch)
        return self.ce_weight * value, {self.CE_TERM: value.detach()}


class RevisionContinuationProbe(A1Probe):
    def __init__(
        self,
        trunk: nn.Module,
        head: nn.Module,
        checkpoint_loss_module: nn.Module,
        *,
        arm: str,
        lambda_kv: float = 0.01,
        m_projections: int = 1024,
        **kwargs: Any,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}; got {arm!r}")
        self.arm = arm
        base = (
            CEOnlyBaseLoss(checkpoint_loss_module)
            if arm == "ce_only"
            else checkpoint_loss_module
        )
        super().__init__(
            trunk, head, base,
            lambda_kv=lambda_kv, m_projections=m_projections, **kwargs,
        )
        n_layers, d_kv = self.bridge.n_layers, self.bridge.d_kv
        if arm == "ce_only":
            self.bridge = NullBridge(n_layers, d_kv).to(self.device)
            self.effective_lambda_kv = 0.0
        else:
            self.bridge = TargetedSIGRegBridge(
                n_layers, d_kv,
                lambda_kv=lambda_kv, m_projections=m_projections,
                targets="k" if arm == "h_k_only" else "v",
            ).to(self.device)
            self.effective_lambda_kv = lambda_kv


def run_revision_continuation(
    trunk: nn.Module,
    head: nn.Module,
    checkpoint_loss_module: nn.Module,
    *,
    arm: str,
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
    on_trained: Callable[[RevisionContinuationProbe], None] | None = None,
    expected_steps: int | None = None,
    aniso_seed: int = 1234,
) -> BridgeProbeResult:
    dev = device or next(trunk.parameters()).device

    geo0 = measure_per_layer_kv_geometry(trunk, eval_loader_factory(), device=dev, seed=aniso_seed)
    nll_before = evaluate(trunk, head, eval_loader_factory(), device=dev)["mean_nll"]
    n_meas = 2 * len(geo0["k_aniso"])
    aniso_before = float(sum(geo0["k_aniso"] + geo0["v_aniso"]) / n_meas)

    probe = RevisionContinuationProbe(
        trunk, head, checkpoint_loss_module,
        arm=arm, lambda_kv=lambda_kv, m_projections=m_projections,
        device=dev, lr=lr, unfreeze=unfreeze, amp_dtype=amp_dtype,
    )
    last = probe.train(train_loader, n_steps=n_steps, on_step=on_step)
    if on_trained is not None:
        on_trained(probe)
    if expected_steps is not None and probe.global_step != expected_steps:
        raise RuntimeError(
            f"run_revision_continuation({arm}): trained {probe.global_step} of "
            f"{expected_steps} expected steps (data stream exhausted?) — "
            "refusing to evaluate/save as a completed run"
        )

    geo1 = measure_per_layer_kv_geometry(trunk, eval_loader_factory(), device=dev, seed=aniso_seed)
    nll_after = evaluate(trunk, head, eval_loader_factory(), device=dev)["mean_nll"]
    aniso_after = float(sum(geo1["k_aniso"] + geo1["v_aniso"]) / n_meas)

    aniso_floor = 0.01
    reliable = aniso_before > aniso_floor
    reduction = (aniso_before - aniso_after) / aniso_before if reliable else 0.0
    nll_delta = nll_after - nll_before

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs))

    return BridgeProbeResult(
        probe=f"revision_{arm}_{unfreeze}",
        n_steps_trained=probe.global_step,
        gate_metric="per_layer_kv_anisotropy_reduction",
        gate_value=reduction,
        gate_threshold=0.30,
        passed=bool(reliable and reduction >= 0.30 and nll_delta <= 0.15),
        per_layer={
            "k_aniso_before": geo0["k_aniso"], "v_aniso_before": geo0["v_aniso"],
            "k_aniso_after": geo1["k_aniso"], "v_aniso_after": geo1["v_aniso"],
            "k_top1_before": geo0["k_top1"], "v_top1_before": geo0["v_top1"],
            "k_top1_after": geo1["k_top1"], "v_top1_after": geo1["v_top1"],
        },
        last_train_loss=last.get("total"),
        extra={
            "arm": arm,
            "base_objective": "ce" if arm == "ce_only" else "checkpoint",
            "effective_lambda_kv": probe.effective_lambda_kv,
            "dropped_terms": (
                probe.base_loss.dropped_terms if arm == "ce_only" else {}
            ),
            "gate_is_informational": True,
            "aniso_before": aniso_before,
            "aniso_after": aniso_after,
            "aniso_before_reliable": reliable,
            "aniso_floor": aniso_floor,
            "spectrum_top1_before": _mean(geo0["k_top1"] + geo0["v_top1"]),
            "spectrum_top1_after": _mean(geo1["k_top1"] + geo1["v_top1"]),
            "nll_before": nll_before,
            "nll_after": nll_after,
            "nll_delta": nll_delta,
            "nll_delta_threshold": 0.15,
        },
    )


def build_adapted_trunk_blob(
    trunk: nn.Module,
    head: nn.Module,
    *,
    trunk_kwargs: Mapping[str, Any],
    tie_lm_head: bool,
    source_checkpoint: str,
    probe_result: Mapping[str, Any],
    arm: str,
    source_seed: int,
    continuation_seed: int,
    recipe: Mapping[str, Any],
    dev_mode: bool = False,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}; got {arm!r}")
    return {
        "format": "a1_adapted_trunk_v1",
        "trunk": trunk.state_dict(),
        "head": head.state_dict(),
        "trunk_kwargs": dict(trunk_kwargs),
        "tie_lm_head": tie_lm_head,
        "source_checkpoint": source_checkpoint,
        "probe_result": dict(probe_result),
        "arm": arm,
        "source_seed": source_seed,
        "continuation_seed": continuation_seed,
        "recipe": dict(recipe),
        "dev_mode": bool(dev_mode),
    }


def verify_source_checkpoint(
    trainer: Any,
    *,
    checkpoint_path: str,
    source_seed: int,
    expect_global_step: int = 610352,
    expect_weights: Mapping[str, float] | None = None,
    expect_sigreg_m: int = 1024,
) -> None:
    embedded = getattr(getattr(trainer, "config", None), "seed", None)
    if embedded is None:
        raise ValueError("source checkpoint config carries no seed — cannot verify identity")
    if int(embedded) != int(source_seed):
        raise ValueError(
            f"source checkpoint's embedded config seed is {embedded}, "
            f"but --source-seed claims {source_seed} (path renames cannot mask this)"
        )
    expect_weights = dict(expect_weights or {"l_aux": 1.0, "l_sigreg": 0.01})
    step = getattr(trainer, "global_step", None)
    if step != expect_global_step:
        raise ValueError(
            f"source checkpoint global_step={step}, expected {expect_global_step}"
        )
    loss = getattr(trainer, "loss_module", None)
    weights = getattr(loss, "weights", None)
    terms = getattr(loss, "terms", None)
    if weights is None or terms is None:
        raise ValueError("source checkpoint loss module is not a JEPALMLoss")
    if {k: float(v) for k, v in weights.items()} != expect_weights:
        raise ValueError(
            f"source checkpoint loss weights {dict(weights)} != expected {expect_weights}"
        )
    ce = terms.get("l_aux") if hasattr(terms, "get") else terms["l_aux"]
    if type(ce).__name__ != "AuxCELoss":
        raise ValueError(f"source checkpoint CE term is {type(ce).__name__}, expected AuxCELoss")
    if getattr(ce, "label_smoothing", 0.0) != 0.0:
        raise ValueError(
            f"source checkpoint CE label_smoothing={ce.label_smoothing} — expected 0.0"
        )
    if getattr(ce, "ignore_index", -100) != -100:
        raise ValueError(
            f"source checkpoint CE ignore_index={ce.ignore_index} — expected -100"
        )
    if getattr(ce, "logits_key", "logits") != "logits" or getattr(ce, "labels_key", "labels") != "labels":
        raise ValueError(
            f"source checkpoint CE reads batch[{getattr(ce, 'logits_key', None)!r}]/"
            f"[{getattr(ce, 'labels_key', None)!r}] — expected 'logits'/'labels'"
        )
    sigreg = terms.get("l_sigreg") if hasattr(terms, "get") else terms["l_sigreg"]
    if getattr(sigreg, "key", None) != "h" or getattr(sigreg, "m_projections", None) != expect_sigreg_m:
        raise ValueError(
            "source checkpoint l_sigreg term is not SIGReg(h) with "
            f"m_projections={expect_sigreg_m} (got key={getattr(sigreg, 'key', None)!r}, "
            f"m={getattr(sigreg, 'm_projections', None)})"
        )
    if f"s{source_seed}" not in checkpoint_path:
        raise ValueError(
            f"checkpoint path {checkpoint_path!r} does not name source seed s{source_seed}"
        )


def claim_out_dir(out_dir: Any) -> None:
    from pathlib import Path

    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"{out} already exists — this runner atomically claims a FRESH "
            "directory; move or delete the previous run's directory first"
        ) from None


def atomic_torch_save(obj: Any, path: Any) -> None:
    import os as _os
    from pathlib import Path

    path = Path(path)
    from jepa_lm.revision.guards import (
        _exclusive_tmp_fd,
        _safe_unlink_tmp,
        _verify_tmp_is_ours,
    )

    tmp, fd = _exclusive_tmp_fd(path)
    try:
        with _os.fdopen(fd, "wb", closefd=False) as f:
            torch.save(obj, f)
            f.flush()
            _os.fsync(fd)
        _verify_tmp_is_ours(tmp, fd)
        _os.replace(tmp, path)
    finally:
        ino = _os.fstat(fd).st_ino
        _os.close(fd)
        _safe_unlink_tmp(tmp, ino)


def atomic_write_text(path: Any, text: str) -> None:
    import os as _os
    from pathlib import Path

    path = Path(path)
    from jepa_lm.revision.guards import (
        _exclusive_tmp_fd,
        _safe_unlink_tmp,
        _verify_tmp_is_ours,
    )

    from jepa_lm.revision.guards import _write_all

    tmp, fd = _exclusive_tmp_fd(path)
    try:
        _write_all(fd, text.encode())
        _os.fsync(fd)
        _verify_tmp_is_ours(tmp, fd)
        _os.replace(tmp, path)
    finally:
        ino = _os.fstat(fd).st_ino
        _os.close(fd)
        _safe_unlink_tmp(tmp, ino)
