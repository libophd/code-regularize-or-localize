from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch

from jepa_lm.data.fineweb import build_fineweb_factories, load_fineweb_streaming
from jepa_lm.training.losses import WeightedTerm
from jepa_lm.training.sweep import SweepResult, run_phase1_sweep
from jepa_lm.training.trainer import TrainerConfig


_AMP_DTYPES: dict[str, torch.dtype | None] = {
    "fp32": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
_DEFAULT_TOKENIZER = "mistralai/Mistral-7B-v0.1"
_DEFAULT_DATASET = "HuggingFaceFW/fineweb"
_DEFAULT_DATASET_CONFIG = "sample-10BT"
_LLAMA_125M_KWARGS: dict[str, int] = dict(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=2048,
    vocab_size=32000,
    max_position_embeddings=2048,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 λ sweep on FineWeb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tokenizer-name", default=_DEFAULT_TOKENIZER,
                   help="HF AutoTokenizer name; must produce ids in [0, vocab_size).")
    p.add_argument("--dataset-name", default=_DEFAULT_DATASET)
    p.add_argument("--dataset-config", default=_DEFAULT_DATASET_CONFIG,
                   help="HF dataset config / subset name (e.g., sample-10BT).")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--eval-docs", type=int, default=5000,
                   help="Held-out documents reserved for eval.")
    p.add_argument("--eval-offset-docs", type=int, default=0,
                   help="Skip this many docs before the eval window; slides eval "
                        "off the head of the stream to mitigate shard/order bias. "
                        "0 is fine for wiring/smoke; set nonzero for gate numbers.")
    p.add_argument("--train-tokens", type=int, required=True,
                   help="Total tokens to train on (sets n_steps via batch_size*seq_len).")
    p.add_argument("--eval-tokens", type=int, required=True,
                   help="Cap on eval tokens (sets eval_n_batches).")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=32000,
                   help="Embedding table size; MUST equal tokenizer vocab_size.")
    p.add_argument("--hidden-size", type=int, default=_LLAMA_125M_KWARGS["hidden_size"])
    p.add_argument("--num-hidden-layers", type=int, default=_LLAMA_125M_KWARGS["num_hidden_layers"])
    p.add_argument("--num-attention-heads", type=int, default=_LLAMA_125M_KWARGS["num_attention_heads"])
    p.add_argument("--intermediate-size", type=int, default=_LLAMA_125M_KWARGS["intermediate_size"])
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0],
                   help="SIGReg λ values (sorted, must be unique); the CE-only "
                        "baseline is arm 0, the K λ values are arms 1..K.")
    p.add_argument("--sigreg-m", type=int, default=1024,
                   help="Number of random projection directions per SIGReg call.")
    p.add_argument("--arm-index", type=int, default=None,
                   help="Run only one sweep arm, for one-arm-per-SLURM-job. "
                        "0 = CE-only baseline; 1..K = the K sorted --lambdas. "
                        "Omit to run the whole sweep in a single process.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42, help="Training seed (reset per arm).")
    p.add_argument("--eval-seed", type=int, default=99,
                   help="Locks eval-side MC RNG across arms (reservoir + anisotropy pairs).")
    p.add_argument("--amp-dtype", choices=list(_AMP_DTYPES.keys()), default="fp32")
    p.add_argument("--device", default=None,
                   help="torch device string (e.g., 'cuda', 'cuda:0'); default = CPU.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint-dir", default=None,
                   help="Run-state root: each arm gets a {label}/ subdir with "
                        "checkpoint_step*.pt and step_metrics.jsonl. "
                        "Default: <out-dir>/checkpoints.")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Write a periodic checkpoint every N steps for mid-arm "
                        "crash recovery. 0 = only the final per-arm checkpoint.")
    p.add_argument("--resume", action="store_true",
                   help="Resume a crashed run: per arm, reload the latest "
                        "checkpoint, skip the consumed train batches, continue. "
                        "Arms with no checkpoint run fresh. Re-run with the same "
                        "--out-dir / --checkpoint-dir as the crashed run.")
    return p.parse_args()


def load_tokenizer_with_vocab_pin(name: str, *, expected_vocab: int):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    vocab = tok.vocab_size
    n_tokens = len(tok)
    eos = tok.eos_token_id

    if vocab != expected_vocab:
        raise ValueError(
            f"tokenizer.vocab_size={vocab} != expected {expected_vocab}. "
            f"Either use a {expected_vocab}-vocab tokenizer or set --vocab-size {vocab}."
        )
    if n_tokens != expected_vocab:
        raise ValueError(
            f"len(tokenizer)={n_tokens} differs from vocab_size={vocab}; "
            f"added-special-tokens drift would land in OOB embedding lookups."
        )
    if eos is None:
        raise ValueError(
            f"tokenizer '{name}' has no eos_token_id; cannot inject document-boundary "
            f"EOS into pack_token_stream. Pick a tokenizer with a defined EOS."
        )
    if eos >= vocab:
        raise ValueError(
            f"tokenizer.eos_token_id={eos} >= vocab_size={vocab}; would land in OOB "
            f"embedding lookup."
        )
    return tok


def git_state(repo_root: Path) -> dict[str, Any]:
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=repo_root, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    sha = _run(["git", "rev-parse", "HEAD"]) or "unknown"
    porcelain = _run(["git", "status", "--porcelain"])
    return {"sha": sha, "dirty": bool(porcelain), "porcelain": porcelain}


def build_audit_snapshot(
    *,
    args: argparse.Namespace,
    cfg: TrainerConfig,
    tokenizer_name: str,
    tokenizer_vocab: int,
    eos_token_id: int,
    n_steps: int,
    n_eval_batches: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    return {
        "trainer_config": _trainer_config_to_dict(cfg),
        "tokenizer": {
            "name": tokenizer_name,
            "vocab_size": tokenizer_vocab,
            "eos_token_id": eos_token_id,
        },
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "eval_docs": args.eval_docs,
            "eval_offset_docs": args.eval_offset_docs,
        },
        "sweep": {
            "lambdas": sorted(args.lambdas),
            "arm_index": args.arm_index,
            "sigreg_m": args.sigreg_m,
            "eval_seed": args.eval_seed,
        },
        "schedule": {
            "n_steps": n_steps,
            "n_eval_batches": n_eval_batches,
            "tokens_per_step": args.batch_size * args.seq_len,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
        },
        "checkpointing": {
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_every": args.checkpoint_every,
            "resume": args.resume,
        },
        "git": git_state(_REPO_ROOT),
    }


def write_audit_artifacts(out_dir: Path, snapshot: dict[str, Any]) -> None:
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (out_dir / "config.json").write_text(json.dumps(snapshot, indent=2, default=str))


def check_resume_audit(prior: dict[str, Any], current: dict[str, Any]) -> None:
    for key in ("dataset", "tokenizer", "sweep", "schedule"):
        old = prior.get(key)
        new = current.get(key)
        if old is not None and old != new:
            raise ValueError(
                f"--resume: '{key}' differs from the prior run recorded in "
                f"config.json:\n  prior:   {old}\n  current: {new}\n"
                f"Resuming would mix two runs. Use a fresh --out-dir, or "
                f"restore the original arguments."
            )


def _trainer_config_to_dict(cfg: TrainerConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["amp_dtype"] = str(cfg.amp_dtype) if cfg.amp_dtype is not None else None
    d["device"] = str(cfg.device) if cfg.device is not None else None
    return d


def _result_to_dict(r: SweepResult) -> dict[str, Any]:
    return {
        "label": r.label,
        "sigreg_weight": r.sigreg_weight,
        "n_steps_trained": r.n_steps_trained,
        "eval_metrics": dict(r.eval_metrics),
        "last_train_step_metrics": dict(r.last_train_step_metrics),
    }


def make_incremental_writer(out_dir: Path) -> Callable[[SweepResult], None]:
    jsonl_path = out_dir / "sweep_metrics.jsonl"
    jsonl_path.write_text("")

    def _on_result(result: SweepResult) -> None:
        with jsonl_path.open("a") as f:
            f.write(json.dumps(_result_to_dict(result), default=str) + "\n")
        print(f"  [incremental] {result.label} -> {jsonl_path.name}")

    return _on_result


def write_sweep_metrics(
    out_dir: Path,
    *,
    results: list[SweepResult],
    elapsed: float,
) -> None:
    payload = {
        "elapsed_seconds": elapsed,
        "results": [_result_to_dict(r) for r in results],
    }
    (out_dir / "sweep_metrics.json").write_text(json.dumps(payload, indent=2, default=str))


def main() -> None:
    args = parse_args()

    lambdas = sorted(args.lambdas)
    if len(lambdas) != len(set(lambdas)):
        dups = sorted({x for x in lambdas if lambdas.count(x) > 1})
        raise SystemExit(
            f"[phase1_real_run] --lambdas has duplicate value(s) {dups}; each "
            f"λ must be unique — duplicates collide on arm labels and per-arm "
            f"checkpoint subdirectories."
        )
    if args.arm_index is None:
        run_sigreg_weights = lambdas
        include_baseline = True
        arm_desc = f"full sweep (baseline + {len(lambdas)} λ)"
    else:
        n_arms = len(lambdas) + 1
        if not 0 <= args.arm_index < n_arms:
            raise SystemExit(
                f"[phase1_real_run] --arm-index {args.arm_index} out of range: "
                f"valid 0..{n_arms - 1} (0=baseline, 1..{len(lambdas)}=sorted "
                f"lambdas {lambdas})"
            )
        if args.arm_index == 0:
            run_sigreg_weights, include_baseline = [], True
            arm_desc = "arm 0 (baseline)"
        else:
            run_sigreg_weights = [lambdas[args.arm_index - 1]]
            include_baseline = False
            arm_desc = f"arm {args.arm_index} (λ={lambdas[args.arm_index - 1]:g})"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "checkpoints"
    )

    print(f"[phase1_real_run] tokenizer={args.tokenizer_name} dataset={args.dataset_name}/{args.dataset_config}")
    tokenizer = load_tokenizer_with_vocab_pin(
        args.tokenizer_name, expected_vocab=args.vocab_size
    )
    print(f"  tokenizer vocab_size={tokenizer.vocab_size} eos_id={tokenizer.eos_token_id}")

    tokens_per_step = args.batch_size * args.seq_len
    n_steps = math.ceil(args.train_tokens / tokens_per_step)
    n_eval_batches = math.ceil(args.eval_tokens / tokens_per_step)
    print(f"  tokens_per_step={tokens_per_step} -> n_steps={n_steps}, n_eval_batches={n_eval_batches}")
    print(f"  selection: {arm_desc}")

    def raw_stream_factory():
        return load_fineweb_streaming(
            args.dataset_name, args.dataset_config, split=args.dataset_split
        )

    def tokenize_fn(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    train_factory, eval_factory = build_fineweb_factories(
        raw_stream_factory,
        tokenize_fn,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        eos_token_id=tokenizer.eos_token_id,
        eval_docs=args.eval_docs,
        eval_offset_docs=args.eval_offset_docs,
        eval_n_batches=n_eval_batches,
    )

    trunk_kwargs = dict(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        max_position_embeddings=max(args.seq_len, _LLAMA_125M_KWARGS["max_position_embeddings"]),
    )
    cfg = TrainerConfig(
        trunk_kwargs=trunk_kwargs,
        tie_lm_head=True,
        loss_terms=[WeightedTerm(name="l_aux", weight=1.0)],
        loss_term_kwargs={},
        n_steps=n_steps,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        lr=args.lr,
        grad_clip=args.grad_clip,
        seed=args.seed,
        amp_dtype=_AMP_DTYPES[args.amp_dtype],
        device=torch.device(args.device) if args.device else None,
    )

    snapshot = build_audit_snapshot(
        args=args,
        cfg=cfg,
        tokenizer_name=args.tokenizer_name,
        tokenizer_vocab=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        n_steps=n_steps,
        n_eval_batches=n_eval_batches,
        checkpoint_dir=checkpoint_dir,
    )
    config_path = out_dir / "config.json"
    if args.resume and config_path.exists():
        check_resume_audit(json.loads(config_path.read_text()), snapshot)
    write_audit_artifacts(out_dir, snapshot)
    print(f"  audit snapshot written to {config_path}")
    print(f"  checkpoints -> {checkpoint_dir}{'  [RESUME]' if args.resume else ''}")

    incremental_writer = make_incremental_writer(out_dir)

    t0 = time.time()
    results = run_phase1_sweep(
        base_config=cfg,
        sigreg_weights=run_sigreg_weights,
        include_baseline=include_baseline,
        train_loader_factory=train_factory,
        eval_loader_factory=eval_factory,
        sigreg_term_kwargs={"m_projections": args.sigreg_m, "key": "h"},
        eval_seed=args.eval_seed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        on_result=incremental_writer,
        verbose=True,
    )
    elapsed = time.time() - t0

    write_sweep_metrics(out_dir, results=results, elapsed=elapsed)
    print(
        f"\n[phase1_real_run] done in {elapsed:.1f}s; "
        f"consolidated metrics at {out_dir / 'sweep_metrics.json'} "
        f"(per-arm log: {out_dir / 'sweep_metrics.jsonl'})"
    )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
