from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import torch

from jepa_lm.data.fineweb import build_fineweb_factories, load_fineweb_streaming
from jepa_lm.systems.a1_probe import run_a1_probe
from jepa_lm.systems.bridge_probe import run_a2_probe, run_a3_probe
from jepa_lm.training.checkpoint import load_checkpoint
from phase1_real_run import load_tokenizer_with_vocab_pin

_AMP_DTYPES: dict[str, torch.dtype | None] = {
    "fp32": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
_DEFAULT_TOKENIZER = "mistralai/Mistral-7B-v0.1"
_DEFAULT_DATASET = "HuggingFaceFW/fineweb"
_DEFAULT_DATASET_CONFIG = "sample-10BT"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1.5 bridge probe on a Phase-1 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--probe", required=True, choices=["a1", "a2", "a3"])
    p.add_argument("--checkpoint", required=True, help="Phase-1 .pt (the λ=0.01 arm).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tokenizer-name", default=_DEFAULT_TOKENIZER)
    p.add_argument("--dataset-name", default=_DEFAULT_DATASET)
    p.add_argument("--dataset-config", default=_DEFAULT_DATASET_CONFIG)
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--eval-docs", type=int, default=5000)
    p.add_argument("--eval-offset-docs", type=int, default=100_000)
    p.add_argument("--train-tokens", type=int, default=500_000_000)
    p.add_argument("--eval-tokens", type=int, default=2_000_000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None,
                   help="Probe LR; default 1e-4 for a1 (continue-train), 1e-3 for a2/a3.")
    p.add_argument("--m-projections", type=int, default=1024,
                   help="SIGReg projection directions (a1 multi-layer SIGReg).")
    p.add_argument("--lambda-kv", type=float, default=0.01, help="a1 multi-layer SIGReg weight.")
    p.add_argument("--unfreeze", choices=["trunk", "kv_proj"], default="trunk",
                   help="a1 only: unfreeze the whole trunk or just W_K/W_V.")
    p.add_argument("--a2-bridge", default="per_layer_kv_predictors",
                   choices=["per_layer_kv_predictors", "per_layer_kv_mlp_predictors"],
                   help="a2 only: predictor family. The pre-registered gate is the "
                        "linear default; the MLP variant is the capacity-check arm.")
    p.add_argument("--a2-mlp-hidden-multiplier", type=int, default=2,
                   help="a2 MLP bridge only: hidden width as a multiple of d_kv.")
    p.add_argument("--amp-dtype", choices=list(_AMP_DTYPES.keys()), default="fp32",
                   help="a1 honours this (bf16 recommended on the 5090); a2/a3 run fp32.")
    p.add_argument("--save-trunk", action="store_true",
                   help="a1 only: save the ADAPTED trunk+head to <out-dir>/adapted_trunk.pt "
                        "after the probe (the downstream KV-quantization payoff experiment "
                        "needs the adapted weights; without this flag they are discarded).")
    p.add_argument("--device", default=None, help="torch device, e.g. 'cuda'.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def git_state(repo_root: Path) -> dict[str, Any]:
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=repo_root, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "sha": _run(["git", "rev-parse", "HEAD"]) or "unknown",
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cpu")
    amp_dtype = _AMP_DTYPES[args.amp_dtype]
    lr = args.lr if args.lr is not None else (1e-4 if args.probe == "a1" else 1e-3)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase1.5] probe={args.probe} checkpoint={args.checkpoint}")
    trainer = load_checkpoint(args.checkpoint)
    trunk, head, base_loss = trainer.trunk, trainer.head, trainer.loss_module
    trunk.to(device)
    head.to(device)
    base_loss.to(device)
    print(f"[phase1.5] loaded trunk d_model={trunk.d_model} vocab={trunk.vocab_size} "
          f"global_step={trainer.global_step}")

    tokenizer = load_tokenizer_with_vocab_pin(args.tokenizer_name, expected_vocab=args.vocab_size)

    tokens_per_step = args.batch_size * args.seq_len
    n_steps = math.ceil(args.train_tokens / tokens_per_step)
    n_eval_batches = math.ceil(args.eval_tokens / tokens_per_step)
    print(f"[phase1.5] tokens_per_step={tokens_per_step} n_steps={n_steps} "
          f"n_eval_batches={n_eval_batches} lr={lr} amp={args.amp_dtype}")

    def raw_stream_factory():
        return load_fineweb_streaming(args.dataset_name, args.dataset_config, split=args.dataset_split)

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

    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (out_dir / "config.json").write_text(json.dumps({
        "probe": args.probe,
        "checkpoint": str(args.checkpoint),
        "trunk": {"d_model": trunk.d_model, "vocab_size": trunk.vocab_size,
                  "global_step": trainer.global_step},
        "schedule": {"n_steps": n_steps, "n_eval_batches": n_eval_batches,
                     "tokens_per_step": tokens_per_step, "batch_size": args.batch_size,
                     "seq_len": args.seq_len, "train_tokens": args.train_tokens},
        "probe_cfg": {"lr": lr, "m_projections": args.m_projections,
                      "lambda_kv": args.lambda_kv,
                      "unfreeze": args.unfreeze if args.probe == "a1" else None,
                      "trunk_policy": args.unfreeze if args.probe == "a1" else "frozen",
                      "a2_bridge": args.a2_bridge if args.probe == "a2" else None,
                      "a2_mlp_hidden_multiplier": (
                          args.a2_mlp_hidden_multiplier
                          if args.probe == "a2" and args.a2_bridge == "per_layer_kv_mlp_predictors"
                          else None
                      ),
                      "amp_dtype": args.amp_dtype, "seed": args.seed},
        "dataset": {"name": args.dataset_name, "config": args.dataset_config,
                    "eval_docs": args.eval_docs, "eval_offset_docs": args.eval_offset_docs},
        "tokenizer": {"name": args.tokenizer_name, "vocab_size": tokenizer.vocab_size},
        "git": git_state(_REPO_ROOT),
    }, indent=2, default=str))

    def _log(step: int, payload: object) -> None:
        if step % 50 == 0:
            print({"step": step, "loss": payload})

    t0 = time.time()
    if args.probe == "a2":
        bridge_kwargs = (
            {"hidden_multiplier": args.a2_mlp_hidden_multiplier}
            if args.a2_bridge == "per_layer_kv_mlp_predictors"
            else {}
        )
        result = run_a2_probe(
            trunk, n_steps=n_steps, train_loader=train_factory(),
            eval_loader=eval_factory(), device=device, lr=lr,
            bridge_name=args.a2_bridge, bridge_kwargs=bridge_kwargs, on_step=_log,
        )
    elif args.probe == "a3":
        result = run_a3_probe(
            trunk, n_steps=n_steps, train_loader=train_factory(),
            eval_loader=eval_factory(), device=device, lr=lr, on_step=_log,
        )
    else:
        result = run_a1_probe(
            trunk, head, base_loss, n_steps=n_steps, train_loader=train_factory(),
            eval_loader_factory=eval_factory, lambda_kv=args.lambda_kv,
            m_projections=args.m_projections, device=device, lr=lr,
            unfreeze=args.unfreeze, amp_dtype=amp_dtype, on_step=_log,
        )
    elapsed = time.time() - t0

    if args.save_trunk:
        if args.probe != "a1":
            raise SystemExit("[phase1.5] --save-trunk only applies to a1 (a2/a3 never modify the trunk).")
        trunk_path = out_dir / "adapted_trunk.pt"
        torch.save({
            "format": "a1_adapted_trunk_v1",
            "trunk": trunk.state_dict(),
            "head": head.state_dict(),
            "trunk_kwargs": trainer.config.trunk_kwargs,
            "tie_lm_head": trainer.config.tie_lm_head,
            "source_checkpoint": str(args.checkpoint),
            "probe_result": result.to_dict(),
        }, trunk_path)
        print(f"[phase1.5] adapted trunk saved -> {trunk_path}")

    payload = result.to_dict()
    payload["elapsed_seconds"] = elapsed
    (out_dir / "probe_result.json").write_text(json.dumps(payload, indent=2, default=str))
    verdict = "PASS" if result.passed else "FAIL"
    print(f"\n[phase1.5] {args.probe} {verdict}  {result.gate_metric}="
          f"{result.gate_value:.4f} (threshold {result.gate_threshold}) "
          f"in {elapsed:.1f}s -> {out_dir / 'probe_result.json'}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
