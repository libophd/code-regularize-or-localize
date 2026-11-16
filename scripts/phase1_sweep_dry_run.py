from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch

from jepa_lm.training.losses import WeightedTerm
from jepa_lm.training.sweep import run_phase1_sweep
from jepa_lm.training.trainer import TrainerConfig


VOCAB = 100


def _synth_batch(*, b: int, t: int, seed: int) -> dict:
    g = torch.Generator()
    g.manual_seed(seed)
    ids = torch.randint(0, VOCAB, (b, t), generator=g)
    return {"input_ids": ids, "labels": ids.clone()}


def _deterministic_loader_factory(n_batches: int, *, b: int = 4, t: int = 16, offset: int = 0):
    def make():
        return [_synth_batch(b=b, t=t, seed=i + offset) for i in range(n_batches)]
    return make


def main() -> None:
    base = TrainerConfig(
        trunk_kwargs=dict(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            vocab_size=VOCAB,
            max_position_embeddings=64,
        ),
        tie_lm_head=True,
        loss_terms=[WeightedTerm(name="l_aux", weight=1.0)],
        loss_term_kwargs={},
        n_steps=5,
        warmup_steps=2,
        log_every=1000,
        lr=3e-4,
        grad_clip=1.0,
        seed=0,
    )

    train_factory = _deterministic_loader_factory(n_batches=8)
    eval_factory = _deterministic_loader_factory(n_batches=4, offset=1000)

    print("\n>>> First sweep invocation")
    r1 = run_phase1_sweep(
        base_config=base,
        sigreg_weights=[0.001, 0.01, 0.1, 1.0],
        train_loader_factory=train_factory,
        eval_loader_factory=eval_factory,
        sigreg_term_kwargs={"m_projections": 64, "key": "h"},
        verbose=True,
    )

    assert len(r1) == 5, f"expected 5 runs (baseline + 4 lambdas); got {len(r1)}"
    for r in r1:
        assert math.isfinite(r.eval_metrics["perplexity"]), r
        assert -1.0 <= r.eval_metrics["anisotropy_h"] <= 1.0, r
        assert 0.0 < r.eval_metrics["spectrum_top1_share"] <= 1.0, r

    print("\n>>> Second sweep invocation (determinism check)")
    r2 = run_phase1_sweep(
        base_config=base,
        sigreg_weights=[0.001, 0.01, 0.1, 1.0],
        train_loader_factory=train_factory,
        eval_loader_factory=eval_factory,
        sigreg_term_kwargs={"m_projections": 64, "key": "h"},
        verbose=False,
    )

    print("\n=== Determinism check ===")
    for a, b in zip(r1, r2):
        delta_ppl = abs(a.eval_metrics["perplexity"] - b.eval_metrics["perplexity"])
        delta_aniso = abs(a.eval_metrics["anisotropy_h"] - b.eval_metrics["anisotropy_h"])
        print(
            f"  {a.label:<12s}  |Δppl|={delta_ppl:.2e}  |Δaniso|={delta_aniso:.2e}"
        )
        assert delta_ppl < 1e-3, (
            f"determinism broken on {a.label}: ppl differs by {delta_ppl}"
        )
        assert delta_aniso < 1e-3, (
            f"determinism broken on {a.label}: aniso differs by {delta_aniso}"
        )

    print("\nSweep dry run: PASS")


if __name__ == "__main__":
    main()
