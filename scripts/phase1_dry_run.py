from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch

from jepa_lm.eval.evaluator import evaluate
from jepa_lm.training.losses import WeightedTerm
from jepa_lm.training.trainer import Trainer, TrainerConfig


def main() -> None:
    torch.manual_seed(0)

    cfg = TrainerConfig(
        trunk_kwargs=dict(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            vocab_size=100,
            max_position_embeddings=64,
        ),
        tie_lm_head=True,
        loss_terms=[
            WeightedTerm(name="l_aux", weight=1.0),
            WeightedTerm(name="l_sigreg", weight=0.1),
        ],
        loss_term_kwargs={"l_sigreg": {"m_projections": 64, "key": "h"}},
        n_steps=10,
        warmup_steps=3,
        log_every=1,
        lr=3e-4,
        grad_clip=1.0,
    )

    print("Building trainer...")
    trainer = Trainer(cfg)
    print(f"  trunk d_model={trainer.trunk.d_model}, vocab={trainer.trunk.vocab_size}")
    print(f"  optimizer params (deduped): {len(trainer.optimizer.param_groups[0]['params'])}")

    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    print(f"  initial LR (LambdaLR-set at construction): {initial_lr:.2e}")
    assert initial_lr > 0.0, (
        f"BUG: initial LR is {initial_lr}; first optimizer.step() would be "
        f"a no-op. Warmup lambda must return >0 at step 0."
    )

    embed_w = trainer.trunk.get_input_embeddings().weight
    head_w = trainer.head.proj.weight
    print(f"  tied: head.proj.weight is embed.weight = {head_w is embed_w}")
    assert head_w is embed_w, "tying did not share the Parameter"

    before = {n: p.detach().clone() for n, p in trainer.trunk.named_parameters()}

    print("\nRunning 5 dry-run steps...")
    for _ in range(5):
        ids = torch.randint(0, 100, (4, 16))
        batch = {"input_ids": ids, "labels": ids.clone()}
        m = trainer.step(batch)
        assert torch.isfinite(torch.tensor(m["loss"])).item(), f"non-finite loss: {m['loss']}"
        print(
            f"  step={m['step']:>2d}  "
            f"loss={m['loss']:.4f}  "
            f"l_aux={m['components']['l_aux']:.4f}  "
            f"l_sigreg={m['components']['l_sigreg']:.4f}  "
            f"lr={m['lr']:.2e}  "
            f"grad_norm={m['grad_norm']:.4f}"
        )

    moved_names = [
        n for n, p in trainer.trunk.named_parameters() if not torch.equal(before[n], p)
    ]
    all_names = [n for n, _ in trainer.trunk.named_parameters()]
    print(f"\nTrunk parameters changed: {len(moved_names)}/{len(all_names)}")
    if len(moved_names) != len(all_names):
        unmoved = [n for n in all_names if n not in moved_names]
        raise AssertionError(
            f"BUG: {len(unmoved)} trunk parameter(s) did not move after 5 steps — "
            f"likely silently disconnected from the gradient path. Unmoved: {unmoved}"
        )

    after_head_w = trainer.head.proj.weight
    after_embed_w = trainer.trunk.get_input_embeddings().weight
    print(f"Tying after 5 steps: head.proj.weight is embed.weight = {after_head_w is after_embed_w}")
    assert after_head_w is after_embed_w, "tying broke during training"

    print("\nRunning eval pass on synthetic held-out loader...")
    eval_loader = [
        {"input_ids": torch.randint(0, 100, (4, 16)), "labels": torch.randint(0, 100, (4, 16))}
        for _ in range(4)
    ]
    metrics = evaluate(
        trainer.trunk,
        trainer.head,
        eval_loader,
        anisotropy_buffer_size=128,
        anisotropy_n_pairs=500,
    )
    print(
        f"  perplexity={metrics['perplexity']:.4f}  "
        f"mean_nll={metrics['mean_nll']:.4f}  "
        f"n_tokens={metrics['n_tokens']}"
    )
    print(
        f"  anisotropy_h={metrics['anisotropy_h']:.4f}  "
        f"top1_spectrum_share={metrics['spectrum_top1_share']:.4f}  "
        f"spec_max={metrics['spectrum_max']:.4f}  "
        f"spec_min={metrics['spectrum_min']:.4f}  "
        f"n_h={metrics['n_h_collected']}"
    )
    import math as _math
    assert _math.isfinite(metrics["perplexity"]) and metrics["perplexity"] > 0
    assert -1.0 <= metrics["anisotropy_h"] <= 1.0
    assert 0.0 < metrics["spectrum_top1_share"] <= 1.0
    assert metrics["n_tokens"] > 0

    print("\nDry run: PASS")


if __name__ == "__main__":
    main()
