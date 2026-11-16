from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import torch

from jepa_lm.data.fineweb import build_fineweb_factories
from jepa_lm.revision.hf_pins import (
    load_pinned_fineweb,
    load_pinned_tokenizer,
    resolve_cached_revision,
)
from jepa_lm.revision.envcheck import (
    TRAIN_FREEZE_REL_PATH,
    TRAIN_FREEZE_SHA256,
    check_freeze_complete,
)
from jepa_lm.revision.guards import enforce_single_visible_gpu
from jepa_lm.revision.sweep_configs import (
    frozen_new_arm_recipes,
    registered_arm1_sha256,
)
from jepa_lm.revision.continuation import (
    ARMS,
    atomic_torch_save,
    atomic_write_text,
    build_adapted_trunk_blob,
    claim_out_dir,
    run_revision_continuation,
    verify_source_checkpoint,
)
from jepa_lm.training.checkpoint import load_checkpoint

_AMP_DTYPES: dict[str, torch.dtype | None] = {
    "fp32": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
_EXPECTED_GLOBAL_STEP = 610352


class _Tee:
    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self._stream, self._log = stream, log

    def write(self, data: str) -> int:
        n = self._stream.write(data)
        self._log.write(data)
        self._log.flush()
        return n

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one revision continuation arm from a Phase-1 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--checkpoint", required=True,
                   help="Phase-1 arm1 (sigreg_l0.01) .pt — the SAME starting point A1 used.")
    p.add_argument("--source-seed", type=int, required=True,
                   help="Seed of the SOURCE checkpoint (42/43/44); verified against the path.")
    p.add_argument("--continuation-seed", type=int, default=42,
                   help="Recorded pre-load manual_seed (historical protocol; the "
                        "effective RNG is checkpoint-restored — see module docstring).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--log-file", default=None,
                   help="Run log path; default outputs/revision/logs/<arm>_s<source_seed>.log")
    p.add_argument("--tokenizer-name", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--eval-docs", type=int, default=5000)
    p.add_argument("--eval-offset-docs", type=int, default=100_000)
    p.add_argument("--train-tokens", type=int, default=500_000_000)
    p.add_argument("--eval-tokens", type=int, default=2_000_000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--m-projections", type=int, default=256,
                   help="Recorded A1-config value (NOT the library default 1024).")
    p.add_argument("--lambda-kv", type=float, default=0.01,
                   help="h_k_only/h_v_only cache-auxiliary weight; ce_only forces 0.")
    p.add_argument("--unfreeze", choices=["trunk", "kv_proj"], default="trunk")
    p.add_argument("--amp-dtype", choices=list(_AMP_DTYPES.keys()), default="bf16")
    p.add_argument("--device", default=None)
    p.add_argument("--allow-off-recipe", action="store_true",
                   help="dev only: skip the frozen-recipe/clean-git pre-training gate")
    return p.parse_args()


def parse_args_for_test(argv: list[str]) -> argparse.Namespace:
    import sys as _sys

    old = _sys.argv
    try:
        _sys.argv = ["run_continuation_arm.py", *argv]
        return parse_args()
    finally:
        _sys.argv = old


def build_recipe(args: argparse.Namespace, *, device_type: str,
                 tokenizer_revision: str, dataset_revision: str,
                 source_sha256: str, git: dict[str, Any]) -> dict[str, Any]:
    return {
        "dev_mode": bool(args.allow_off_recipe),
        "lr": args.lr,
        "lambda_kv": 0.0 if args.arm == "ce_only" else args.lambda_kv,
        "m_projections": args.m_projections,
        "amp_dtype": args.amp_dtype,
        "unfreeze": args.unfreeze,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "train_tokens": args.train_tokens,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "device_type": device_type,
        "rng_policy": "checkpoint_restored",
        "tokenizer_name": args.tokenizer_name,
        "vocab_size": args.vocab_size,
        "dataset_split": args.dataset_split,
        "eval_tokens": args.eval_tokens,
        "eval_docs": args.eval_docs,
        "eval_offset_docs": args.eval_offset_docs,
        "aniso_seed": 1234,
        "git_dirty": git["dirty"],
        "tokenizer_revision": tokenizer_revision,
        "dataset_revision": dataset_revision,
        "train_freeze_sha256": TRAIN_FREEZE_SHA256,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "git_sha": git["sha"],
    }


def git_state(repo_root: Path) -> dict[str, Any]:
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=repo_root, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise SystemExit(
                f"[revision] git command {' '.join(cmd)} failed ({e}) — "
                "refusing to run with unknown code identity") from None

    sha = _run(["git", "rev-parse", "HEAD"])
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise SystemExit(f"[revision] git sha {sha!r} is not 40-hex — aborting")
    dirty = bool(_run(["git", "status", "--porcelain"]))
    return {"sha": sha, "dirty": dirty}


def main() -> None:
    args = parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = Path(
        args.log_file
        or _REPO_ROOT / "outputs" / "revision" / "logs"
        / f"{args.arm}_s{args.source_seed}_{stamp}_p{os.getpid()}.log"
    )
    out_dir = Path(args.out_dir)
    from jepa_lm.revision.sweep_configs import (
        REGISTERED_CONTINUATION_OUT_ROOT,
        REVISION_OUT_ROOT,
        assert_isolated_out_path,
        assert_legacy_trees_locked,
    )
    assert_isolated_out_path(
        log_path, _REPO_ROOT, kind="continuation log",
        required_root=None if args.allow_off_recipe else REVISION_OUT_ROOT)
    lp, od = log_path.resolve(), out_dir.resolve()
    if lp == od or od in lp.parents or lp in od.parents:
        raise SystemExit(
            f"[revision] --log-file {log_path} overlaps --out-dir {out_dir} "
            "(at/inside it, or an ancestor of it) — the log would break the "
            "fresh-directory claim; keep logs under outputs/revision/logs/")

    def _recheck_out_dir() -> None:
        assert_isolated_out_path(
            out_dir, _REPO_ROOT, kind="continuation out-dir",
            required_root=(None if args.allow_off_recipe
                           else REGISTERED_CONTINUATION_OUT_ROOT),
            forbidden_root=(REGISTERED_CONTINUATION_OUT_ROOT
                            if args.allow_off_recipe else None))

    _recheck_out_dir()
    if not args.allow_off_recipe:
        if (_REPO_ROOT / "outputs" / "kv_quant").exists():
            assert_legacy_trees_locked(_REPO_ROOT)
        else:
            print("[revision] no local paper record on this host (deploy "
                  "node) — legacy-lock walk skipped; source-checkpoint "
                  "digest verification still applies")

    assert_isolated_out_path(
        log_path, _REPO_ROOT, kind="continuation log",
        required_root=None if args.allow_off_recipe else REVISION_OUT_ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("x")
    sys.stdout = _Tee(sys.__stdout__, log)
    sys.stderr = _Tee(sys.__stderr__, log)
    print(f"[revision] log -> {log_path}")

    _recheck_out_dir()
    claim_out_dir(out_dir)

    torch.manual_seed(args.continuation_seed)
    device = torch.device(args.device) if args.device else torch.device("cpu")
    enforce_single_visible_gpu(device)
    amp_dtype = _AMP_DTYPES[args.amp_dtype]

    print(f"[revision] arm={args.arm} source_seed={args.source_seed} "
          f"continuation_seed={args.continuation_seed} checkpoint={args.checkpoint}")
    git0 = git_state(_REPO_ROOT)
    tokenizer_revision = resolve_cached_revision("models", args.tokenizer_name)
    dataset_revision = resolve_cached_revision("datasets", args.dataset_name)
    import hashlib
    h = hashlib.sha256()
    with open(args.checkpoint, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    source_sha256 = h.hexdigest()
    print(f"[revision] source sha256 {source_sha256}")
    if not args.allow_off_recipe:
        check_freeze_complete(_REPO_ROOT, rel_path=TRAIN_FREEZE_REL_PATH,
                              expected_sha256=TRAIN_FREEZE_SHA256)
        want_src = registered_arm1_sha256(args.source_seed)
        if source_sha256 != want_src:
            raise SystemExit(
                f"[revision] source checkpoint sha256 {source_sha256[:16]}… is not "
                f"the registered arm1 checkpoint for seed s{args.source_seed}")
        frozen = frozen_new_arm_recipes()[args.arm]
        recipe = build_recipe(
            args, device_type=device.type,
            tokenizer_revision=tokenizer_revision,
            dataset_revision=dataset_revision,
            source_sha256=source_sha256, git=git0,
        )
        deviations = {k: (recipe.get(k), v) for k, v in frozen.items()
                      if recipe.get(k) != v}
        if args.continuation_seed != 42:
            deviations["continuation_seed"] = (args.continuation_seed, 42)
        if deviations:
            raise SystemExit(
                f"[revision] off-recipe launch (got, frozen): {deviations} — "
                "fix the CLI/checkout or pass --allow-off-recipe for a dev run")
    else:
        recipe = build_recipe(
            args, device_type=device.type,
            tokenizer_revision=tokenizer_revision,
            dataset_revision=dataset_revision,
            source_sha256=source_sha256, git=git0,
        )
    trainer = load_checkpoint(args.checkpoint)
    h2 = hashlib.sha256()
    with open(args.checkpoint, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h2.update(chunk)
    if h2.hexdigest() != source_sha256:
        raise SystemExit(
            "[revision] source checkpoint changed between hashing and loading — aborting")
    verify_source_checkpoint(
        trainer, checkpoint_path=str(args.checkpoint),
        source_seed=args.source_seed, expect_global_step=_EXPECTED_GLOBAL_STEP,
    )
    trunk, head, ckpt_loss = trainer.trunk, trainer.head, trainer.loss_module
    trunk.to(device)
    head.to(device)
    ckpt_loss.to(device)
    print(f"[revision] source verified: d_model={trunk.d_model} vocab={trunk.vocab_size} "
          f"global_step={trainer.global_step}")

    tokenizer = load_pinned_tokenizer(args.tokenizer_name, tokenizer_revision,
                                      expected_vocab=args.vocab_size)

    tokens_per_step = args.batch_size * args.seq_len
    n_steps = math.ceil(args.train_tokens / tokens_per_step)
    n_eval_batches = math.ceil(args.eval_tokens / tokens_per_step)
    print(f"[revision] tokens_per_step={tokens_per_step} n_steps={n_steps} "
          f"n_eval_batches={n_eval_batches} lr={args.lr} amp={args.amp_dtype}")

    def raw_stream_factory():
        return load_pinned_fineweb(args.dataset_name, args.dataset_config,
                                   args.dataset_split, dataset_revision)

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

    _recheck_out_dir()
    atomic_write_text(out_dir / "command.txt", " ".join(sys.argv) + "\n")
    _recheck_out_dir()
    atomic_write_text(out_dir / "config.json", json.dumps({
        "probe": "revision_continuation",
        "dev_mode": bool(args.allow_off_recipe),
        "arm": args.arm,
        "base_objective": "ce" if args.arm == "ce_only" else "checkpoint",
        "checkpoint": str(args.checkpoint),
        "source_seed": args.source_seed,
        "continuation_seed": args.continuation_seed,
        "eval_seed": None,
        "aniso_seed": 1234,
        "log_file": str(log_path),
        "tokenizer_revision": tokenizer_revision,
        "dataset_revision": dataset_revision,
        "source_checkpoint_sha256": source_sha256,
        "rng_policy": "checkpoint_restored, then advanced by the baseline evaluation, "
                      "then training (manual_seed(continuation_seed) pre-load is "
                      "overwritten by load_checkpoint's RNG restore — matches every "
                      "historical A1/continuation run; run with exactly one visible "
                      "GPU so per-device CUDA RNG restore maps 1:1)",
        "trunk": {"d_model": trunk.d_model, "vocab_size": trunk.vocab_size,
                  "global_step": trainer.global_step},
        "schedule": {"n_steps": n_steps, "n_eval_batches": n_eval_batches,
                     "tokens_per_step": tokens_per_step, "batch_size": args.batch_size,
                     "seq_len": args.seq_len, "train_tokens": args.train_tokens},
        "probe_cfg": {"lr": args.lr, "m_projections": args.m_projections,
                      "lambda_kv": 0.0 if args.arm == "ce_only" else args.lambda_kv,
                      "bridge_targets": {"ce_only": "none", "h_k_only": "k",
                                         "h_v_only": "v"}[args.arm],
                      "unfreeze": args.unfreeze, "trunk_policy": args.unfreeze,
                      "amp_dtype": args.amp_dtype, "device": str(device)},
        "dataset": {"name": args.dataset_name, "config": args.dataset_config,
                    "eval_docs": args.eval_docs, "eval_offset_docs": args.eval_offset_docs},
        "tokenizer": {"name": args.tokenizer_name, "vocab_size": tokenizer.vocab_size},
        "git": git0,
    }, indent=2, default=str))

    def _log_step(step: int, payload: object) -> None:
        if step % 50 == 0:
            print({"step": step, "loss": payload})

    partial_path = out_dir / "adapted_trunk.partial.pt"

    def _save_partial(probe) -> None:
        _recheck_out_dir()
        atomic_torch_save(build_adapted_trunk_blob(
            probe.trunk, probe.head,
            trunk_kwargs=trainer.config.trunk_kwargs,
            tie_lm_head=trainer.config.tie_lm_head,
            source_checkpoint=str(args.checkpoint),
            probe_result={
                "status": ("trained_pre_eval"
                           if probe.global_step == n_steps else "incomplete_training"),
                "steps": probe.global_step, "expected_steps": n_steps,
            },
            arm=args.arm, source_seed=args.source_seed,
            continuation_seed=args.continuation_seed, recipe=recipe,
            dev_mode=args.allow_off_recipe,
        ), partial_path)
        print(f"[revision] pre-eval trunk snapshot -> {partial_path} "
              f"(steps={probe.global_step})")

    t0 = time.time()
    result = run_revision_continuation(
        trunk, head, ckpt_loss,
        arm=args.arm, n_steps=n_steps, train_loader=train_factory(),
        eval_loader_factory=eval_factory, lambda_kv=args.lambda_kv,
        m_projections=args.m_projections, device=device, lr=args.lr,
        unfreeze=args.unfreeze, amp_dtype=amp_dtype, on_step=_log_step,
        on_trained=_save_partial, expected_steps=n_steps,
    )
    elapsed = time.time() - t0

    git1 = git_state(_REPO_ROOT)
    if git1 != git0:
        raise SystemExit(
            f"[revision] checkout changed during training ({git0} -> {git1}) — "
            "refusing the final save (partial snapshot retained for forensics)")
    _recheck_out_dir()
    atomic_torch_save(build_adapted_trunk_blob(
        trunk, head,
        trunk_kwargs=trainer.config.trunk_kwargs,
        tie_lm_head=trainer.config.tie_lm_head,
        source_checkpoint=str(args.checkpoint),
        probe_result=result.to_dict(),
        arm=args.arm,
        source_seed=args.source_seed,
        continuation_seed=args.continuation_seed,
        recipe=recipe,
        dev_mode=args.allow_off_recipe,
    ), out_dir / "adapted_trunk.pt")
    _recheck_out_dir()
    partial_path.unlink(missing_ok=True)
    print(f"[revision] adapted trunk saved -> {out_dir / 'adapted_trunk.pt'}")

    payload = result.to_dict()
    payload["elapsed_seconds"] = elapsed
    _recheck_out_dir()
    atomic_write_text(out_dir / "probe_result.json", json.dumps(payload, indent=2, default=str))
    print(f"\n[revision] {args.arm} done  {result.gate_metric}={result.gate_value:.4f} "
          f"(gate informational for control arms) nll_delta="
          f"{result.extra['nll_delta']:+.4f} in {elapsed:.1f}s -> "
          f"{out_dir / 'probe_result.json'}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
