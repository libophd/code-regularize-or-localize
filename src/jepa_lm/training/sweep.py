from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from jepa_lm.eval.evaluator import evaluate
from jepa_lm.training.checkpoint import load_checkpoint, save_checkpoint
from jepa_lm.training.losses import WeightedTerm
from jepa_lm.training.trainer import Trainer, TrainerConfig


@dataclass
class SweepResult:
    label: str
    sigreg_weight: float | None
    eval_metrics: dict[str, Any]
    n_steps_trained: int
    last_train_step_metrics: dict[str, Any] = field(default_factory=dict)


def _build_loss_terms(
    *, aux_weight: float, sigreg_weight: float | None
) -> list[WeightedTerm]:
    terms = [WeightedTerm(name="l_aux", weight=aux_weight)]
    if sigreg_weight is not None:
        terms.append(WeightedTerm(name="l_sigreg", weight=sigreg_weight))
    return terms


def _build_run_config(
    base_config: TrainerConfig,
    *,
    aux_weight: float,
    sigreg_weight: float | None,
    sigreg_term_kwargs: Mapping[str, Any],
) -> TrainerConfig:
    loss_terms = _build_loss_terms(aux_weight=aux_weight, sigreg_weight=sigreg_weight)
    term_kwargs: dict[str, Mapping[str, Any]] = dict(base_config.loss_term_kwargs)
    if sigreg_weight is not None and "l_sigreg" not in term_kwargs:
        term_kwargs["l_sigreg"] = dict(sigreg_term_kwargs)
    return replace(base_config, loss_terms=loss_terms, loss_term_kwargs=term_kwargs)


def run_phase1_sweep(
    *,
    base_config: TrainerConfig,
    sigreg_weights: list[float],
    include_baseline: bool = True,
    train_loader_factory: Callable[[], Iterable[Mapping[str, Tensor]]],
    eval_loader_factory: Callable[[], Iterable[Mapping[str, Tensor]]],
    aux_weight: float = 1.0,
    sigreg_term_kwargs: Mapping[str, Any] | None = None,
    eval_seed: int | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 0,
    resume: bool = False,
    on_result: Callable[[SweepResult], None] | None = None,
    verbose: bool = True,
) -> list[SweepResult]:
    sigreg_term_kwargs = sigreg_term_kwargs or {"key": "h"}

    if checkpoint_every < 0:
        raise ValueError(f"checkpoint_every must be >= 0; got {checkpoint_every}")
    if checkpoint_every > 0 and checkpoint_dir is None:
        raise ValueError("checkpoint_every > 0 requires checkpoint_dir")
    if resume and checkpoint_dir is None:
        raise ValueError("resume=True requires checkpoint_dir")
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else None

    configurations: list[tuple[float | None, str]] = [
        (w, f"sigreg_l{w}") for w in sigreg_weights
    ]
    if include_baseline:
        configurations.insert(0, (None, "baseline"))
    if not configurations:
        raise ValueError(
            "run_phase1_sweep: nothing to run — include_baseline=False and "
            "sigreg_weights is empty."
        )

    results: list[SweepResult] = []
    for sigreg_weight, label in configurations:
        arm_dir = ckpt_root / label if ckpt_root is not None else None
        cfg = _build_run_config(
            base_config,
            aux_weight=aux_weight,
            sigreg_weight=sigreg_weight,
            sigreg_term_kwargs=sigreg_term_kwargs,
        )
        latest_ckpt = _latest_checkpoint(arm_dir) if resume else None

        if latest_ckpt is not None:
            trainer = load_checkpoint(latest_ckpt)
            _check_resume_config(trainer.config, cfg, label, latest_ckpt)
            resume_step = trainer.global_step
            if verbose:
                print(
                    f"\n=== Run: {label} (sigreg_weight={sigreg_weight}) "
                    f"— RESUME from step {resume_step} ==="
                )
        else:
            if verbose:
                print(f"\n=== Run: {label} (sigreg_weight={sigreg_weight}) ===")
            torch.manual_seed(base_config.seed)
            trainer = Trainer(cfg)
            resume_step = 0

        if resume_step >= trainer.config.n_steps:
            if verbose:
                print(f"  -> arm already complete at step {resume_step}; re-evaluating only")
            last_step_metrics: dict[str, Any] = {}
        else:
            train_iter = _skip_consumed_batches(
                iter(train_loader_factory()), resume_step, label
            )
            on_step = (
                _make_step_appender(arm_dir, resume_step=resume_step)
                if arm_dir is not None
                else None
            )
            last_step_metrics = trainer.train(
                train_iter,
                checkpoint_dir=arm_dir,
                checkpoint_every=checkpoint_every,
                on_step=on_step,
                verbose=verbose,
            )
            if arm_dir is not None:
                save_checkpoint(
                    trainer, arm_dir / f"checkpoint_step{trainer.global_step}.pt"
                )

        eval_loader = eval_loader_factory()
        if eval_seed is not None:
            torch.manual_seed(eval_seed)
        eval_metrics = evaluate(trainer.trunk, trainer.head, eval_loader)
        if verbose:
            print(
                f"  -> ppl={eval_metrics['perplexity']:.4f}  "
                f"aniso={eval_metrics['anisotropy_h']:+.4f}  "
                f"top1_share={eval_metrics['spectrum_top1_share']:.4f}"
            )

        result = SweepResult(
            label=label,
            sigreg_weight=sigreg_weight,
            eval_metrics=eval_metrics,
            n_steps_trained=trainer.global_step,
            last_train_step_metrics=last_step_metrics,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)

    if verbose:
        _print_summary_table(results)

    return results


def _print_summary_table(results: list[SweepResult]) -> None:
    headers = ["label", "λ", "ppl", "aniso", "top1_share", "steps"]
    rows = []
    for r in results:
        rows.append([
            r.label,
            "—" if r.sigreg_weight is None else f"{r.sigreg_weight:g}",
            f"{r.eval_metrics['perplexity']:.4f}",
            f"{r.eval_metrics['anisotropy_h']:+.4f}",
            f"{r.eval_metrics['spectrum_top1_share']:.4f}",
            str(r.n_steps_trained),
        ])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    print("\n=== Sweep summary ===")
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))


def _latest_checkpoint(arm_dir: Path | None) -> Path | None:
    if arm_dir is None or not arm_dir.is_dir():
        return None
    best_path: Path | None = None
    best_step = -1
    for p in arm_dir.glob("checkpoint_step*.pt"):
        m = re.fullmatch(r"checkpoint_step(\d+)\.pt", p.name)
        if m is None:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step, best_path = step, p
    return best_path


def _skip_consumed_batches(
    train_iter: Iterator[Mapping[str, Tensor]], n_skip: int, label: str
) -> Iterator[Mapping[str, Tensor]]:
    for i in range(n_skip):
        try:
            next(train_iter)
        except StopIteration:
            raise RuntimeError(
                f"resume: train loader for arm '{label}' ran dry after {i} "
                f"batches — fewer than the checkpoint's global_step {n_skip}. "
                f"The stream is shorter than, or not, the one that produced "
                f"the checkpoint."
            ) from None
    return train_iter


def _short_repr(value: Any, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _check_resume_config(
    ckpt_config: TrainerConfig,
    expected: TrainerConfig,
    label: str,
    ckpt_path: Path,
) -> None:
    if ckpt_config == expected:
        return
    diffs = [
        f"  {f.name}: checkpoint={_short_repr(getattr(ckpt_config, f.name))} "
        f"!= requested={_short_repr(getattr(expected, f.name))}"
        for f in fields(TrainerConfig)
        if getattr(ckpt_config, f.name) != getattr(expected, f.name)
    ]
    raise ValueError(
        f"resume: checkpoint for arm '{label}' ({ckpt_path}) was written "
        f"with a TrainerConfig that differs from this invocation's:\n"
        + "\n".join(diffs)
        + "\nResuming would mix two runs. Use a fresh checkpoint_dir, or "
        "restore the original config."
    )


def _make_step_appender(
    arm_dir: Path, *, resume_step: int
) -> Callable[[dict[str, Any]], None]:
    arm_dir.mkdir(parents=True, exist_ok=True)
    path = arm_dir / "step_metrics.jsonl"
    _trim_step_log(path, resume_step)

    def _append(metrics: dict[str, Any]) -> None:
        with path.open("a") as f:
            f.write(json.dumps(metrics, default=str) + "\n")

    return _append


def _trim_step_log(path: Path, resume_step: int) -> None:
    if resume_step <= 0:
        path.write_text("")
        return
    if not path.exists():
        return
    kept: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("step", 0) <= resume_step:
            kept.append(line)
    path.write_text("".join(f"{ln}\n" for ln in kept))
