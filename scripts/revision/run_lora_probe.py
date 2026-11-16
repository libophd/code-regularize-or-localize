from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from jepa_lm.data.fineweb import build_fineweb_factories
from jepa_lm.revision.continuation import (
    atomic_torch_save,
    atomic_write_text,
    claim_out_dir,
    verify_source_checkpoint,
)
from jepa_lm.revision.envcheck import (
    TRAIN_FREEZE_REL_PATH,
    TRAIN_FREEZE_SHA256,
    check_freeze_complete,
)
from jepa_lm.revision.guards import enforce_single_visible_gpu
from jepa_lm.revision.hf_pins import (
    load_pinned_fineweb,
    load_pinned_tokenizer,
    resolve_cached_revision,
)
from jepa_lm.revision.lora_probe import (
    LORA_KV_RECIPE,
    PROBE_NAME,
    PROBE_OUT_ROOT,
    build_lora_probe_blob,
    code_identity,
    code_identity_unchanged,
    recipe_deviations,
    spec_from_recipe,
)
from jepa_lm.revision.sweep_configs import (
    REGISTERED_CONTINUATION_OUT_ROOT,
    REVISION_OUT_ROOT,
    assert_isolated_out_path,
    assert_legacy_trees_locked,
    registered_arm1_sha256,
)
from jepa_lm.systems.a1_probe import run_a1_probe
from jepa_lm.systems.lora import lora_modules, lora_state_dict, merge_lora
from jepa_lm.training.checkpoint import load_checkpoint

_AMP_DTYPES: dict[str, torch.dtype | None] = {
    "fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16,
}
_EXPECTED_GLOBAL_STEP = 610352
_DEFAULT_CHECKPOINT = ("outputs/phase1_seedconfirm_s42/checkpoints/arm1/sigreg_l0.01/"
                       "checkpoint_step610352.pt")


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    R = LORA_KV_RECIPE
    p = argparse.ArgumentParser(
        description="Run the exploratory LoRA-scale probe (lora_kv) from the registered arm1 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT)
    p.add_argument("--source-seed", type=int, default=R["source_seed"])
    p.add_argument("--continuation-seed", type=int, default=R["continuation_seed"])
    p.add_argument("--out-dir", default=f"{PROBE_OUT_ROOT}/{PROBE_NAME}/s{R['source_seed']}")
    p.add_argument("--log-file", default=None)
    p.add_argument("--tokenizer-name", default=R["tokenizer_name"])
    p.add_argument("--dataset-name", default=R["dataset_name"])
    p.add_argument("--dataset-config", default=R["dataset_config"])
    p.add_argument("--dataset-split", default=R["dataset_split"])
    p.add_argument("--vocab-size", type=int, default=R["vocab_size"])
    p.add_argument("--eval-docs", type=int, default=R["eval_docs"])
    p.add_argument("--eval-offset-docs", type=int, default=R["eval_offset_docs"])
    p.add_argument("--train-tokens", type=int, default=R["train_tokens"])
    p.add_argument("--eval-tokens", type=int, default=R["eval_tokens"])
    p.add_argument("--seq-len", type=int, default=R["seq_len"])
    p.add_argument("--batch-size", type=int, default=R["batch_size"])
    p.add_argument("--lr", type=float, default=R["lr"])
    p.add_argument("--m-projections", type=int, default=R["m_projections"])
    p.add_argument("--lambda-kv", type=float, default=R["lambda_kv"])
    p.add_argument("--rank", type=int, default=R["lora"]["rank"])
    p.add_argument("--alpha", type=float, default=R["lora"]["alpha"])
    p.add_argument("--dropout", type=float, default=R["lora"]["dropout"])
    p.add_argument("--targets", default=",".join(R["lora"]["targets"]))
    p.add_argument("--amp-dtype", choices=list(_AMP_DTYPES), default=R["amp_dtype"])
    p.add_argument("--device", default=None)
    p.add_argument("--dev", action="store_true",
                   help="off-recipe smoke run: skips the frozen-recipe/env/clean-code gate; "
                        "out-dir must be a dev_* directory; artifacts are tagged dev_mode")
    return p.parse_args(argv)


def build_recipe(args: argparse.Namespace, *, device_type: str, tokenizer_revision: str,
                 dataset_revision: str, source_sha256: str, code: dict[str, Any]) -> dict[str, Any]:
    from jepa_lm.systems.lora import LoraSpec
    spec = LoraSpec(rank=args.rank, alpha=args.alpha, dropout=args.dropout,
                    targets=tuple(t for t in args.targets.split(",") if t))
    return {
        "lr": args.lr, "lambda_kv": args.lambda_kv, "m_projections": args.m_projections,
        "amp_dtype": args.amp_dtype, "unfreeze": "lora", "batch_size": args.batch_size,
        "seq_len": args.seq_len, "train_tokens": args.train_tokens,
        "eval_tokens": args.eval_tokens, "eval_docs": args.eval_docs,
        "eval_offset_docs": args.eval_offset_docs, "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config, "dataset_split": args.dataset_split,
        "tokenizer_name": args.tokenizer_name, "vocab_size": args.vocab_size,
        "aniso_seed": 1234, "continuation_seed": args.continuation_seed,
        "source_seed": args.source_seed, "lora": spec.to_dict(),
        "dev_mode": bool(args.dev), "device_type": device_type,
        "rng_policy": "checkpoint_restored",
        "tokenizer_revision": tokenizer_revision, "dataset_revision": dataset_revision,
        "train_freeze_sha256": TRAIN_FREEZE_SHA256,
        "source_checkpoint": str(args.checkpoint), "source_checkpoint_sha256": source_sha256,
        "git_sha": code["sha"], "code_dirty": code["code_dirty"], "tree_dirty": code["tree_dirty"],
    }


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = Path(args.log_file or _REPO_ROOT / "outputs" / "revision" / "logs"
                    / f"{PROBE_NAME}_s{args.source_seed}_{stamp}_p{os.getpid()}.log")
    out_dir = Path(args.out_dir)

    assert_isolated_out_path(log_path, _REPO_ROOT, kind="lora_kv log", required_root=REVISION_OUT_ROOT)
    assert_isolated_out_path(out_dir, _REPO_ROOT, kind="lora_kv out-dir",
                             required_root=PROBE_OUT_ROOT,
                             forbidden_root=REGISTERED_CONTINUATION_OUT_ROOT)
    is_dev_dir = any(part.startswith("dev_") for part in out_dir.resolve().parts)
    if args.dev and not is_dev_dir:
        raise SystemExit("[lora_kv] --dev runs must write under a dev_* directory")
    if not args.dev and is_dev_dir:
        raise SystemExit("[lora_kv] a non-dev run may not write under a dev_* directory")
    lp, od = log_path.resolve(), out_dir.resolve()
    if lp == od or od in lp.parents or lp in od.parents:
        raise SystemExit("[lora_kv] --log-file overlaps --out-dir; keep logs under outputs/revision/logs/")
    if not args.dev and (_REPO_ROOT / "outputs" / "kv_quant").exists():
        assert_legacy_trees_locked(_REPO_ROOT)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("x")
    sys.stdout = _Tee(sys.__stdout__, log)
    sys.stderr = _Tee(sys.__stderr__, log)
    print(f"[lora_kv] log -> {log_path}")
    claim_out_dir(out_dir)

    torch.manual_seed(args.continuation_seed)
    device = torch.device(args.device) if args.device else torch.device("cpu")
    enforce_single_visible_gpu(device)
    amp_dtype = _AMP_DTYPES[args.amp_dtype]
    code0 = code_identity(_REPO_ROOT)
    tokenizer_revision = resolve_cached_revision("models", args.tokenizer_name)
    dataset_revision = resolve_cached_revision("datasets", args.dataset_name)
    source_sha256 = _sha256(args.checkpoint)
    recipe = build_recipe(args, device_type=device.type, tokenizer_revision=tokenizer_revision,
                          dataset_revision=dataset_revision, source_sha256=source_sha256, code=code0)
    print(f"[lora_kv] source sha256 {source_sha256}  code {code0}")

    if not args.dev:
        check_freeze_complete(_REPO_ROOT, rel_path=TRAIN_FREEZE_REL_PATH,
                              expected_sha256=TRAIN_FREEZE_SHA256)
        if source_sha256 != registered_arm1_sha256(args.source_seed):
            raise SystemExit(f"[lora_kv] source checkpoint {source_sha256[:16]}… is not the "
                             f"registered arm1 checkpoint for seed s{args.source_seed}")
        deviations = recipe_deviations(recipe)
        if deviations:
            raise SystemExit(f"[lora_kv] off-recipe launch (got, frozen): {deviations} — "
                             "fix the CLI or pass --dev for a smoke run")
        if code0["code_dirty"]:
            raise SystemExit("[lora_kv] training code paths are dirty — commit first")

    trainer = load_checkpoint(args.checkpoint)
    if _sha256(args.checkpoint) != source_sha256:
        raise SystemExit("[lora_kv] source checkpoint changed between hashing and loading — aborting")
    verify_source_checkpoint(trainer, checkpoint_path=str(args.checkpoint),
                             source_seed=args.source_seed, expect_global_step=_EXPECTED_GLOBAL_STEP)
    trunk, head, ckpt_loss = trainer.trunk, trainer.head, trainer.loss_module
    trunk.to(device)
    head.to(device)
    ckpt_loss.to(device)
    n_base = sum(p.numel() for p in {id(p): p for p in trunk.parameters()}.values())
    print(f"[lora_kv] source verified: d_model={trunk.d_model} vocab={trunk.vocab_size} "
          f"global_step={trainer.global_step} trunk_params={n_base}")

    tokenizer = load_pinned_tokenizer(args.tokenizer_name, tokenizer_revision,
                                      expected_vocab=args.vocab_size)
    tokens_per_step = args.batch_size * args.seq_len
    n_steps = math.ceil(args.train_tokens / tokens_per_step)
    n_eval_batches = math.ceil(args.eval_tokens / tokens_per_step)
    print(f"[lora_kv] tokens_per_step={tokens_per_step} n_steps={n_steps} "
          f"n_eval_batches={n_eval_batches} lr={args.lr} amp={args.amp_dtype} lora={recipe['lora']}")

    def raw_stream_factory():
        return load_pinned_fineweb(args.dataset_name, args.dataset_config,
                                   args.dataset_split, dataset_revision)

    def tokenize_fn(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    train_factory, eval_factory = build_fineweb_factories(
        raw_stream_factory, tokenize_fn, seq_len=args.seq_len, batch_size=args.batch_size,
        eos_token_id=tokenizer.eos_token_id, eval_docs=args.eval_docs,
        eval_offset_docs=args.eval_offset_docs, eval_n_batches=n_eval_batches,
    )
    spec = spec_from_recipe(recipe)
    d = trunk.d_model
    n_lora_expected = len(spec.targets) * trainer.config.trunk_kwargs.get("num_hidden_layers", 12) \
        * spec.rank * (2 * d)

    atomic_write_text(out_dir / "command.txt", " ".join(sys.argv) + "\n")
    atomic_write_text(out_dir / "config.json", json.dumps({
        "probe": PROBE_NAME, "dev_mode": bool(args.dev), "arm": PROBE_NAME,
        "base_objective": "checkpoint", "checkpoint": str(args.checkpoint),
        "source_seed": args.source_seed, "continuation_seed": args.continuation_seed,
        "eval_seed": None, "aniso_seed": 1234, "log_file": str(log_path),
        "tokenizer_revision": tokenizer_revision, "dataset_revision": dataset_revision,
        "source_checkpoint_sha256": source_sha256,
        "rng_policy": "checkpoint_restored, then advanced by the baseline evaluation, then "
                      "training (manual_seed(continuation_seed) pre-load is overwritten by "
                      "load_checkpoint's RNG restore — matches every historical A1/continuation "
                      "run; exactly one visible GPU)",
        "trunk": {"d_model": trunk.d_model, "vocab_size": trunk.vocab_size,
                  "global_step": trainer.global_step, "n_params": n_base},
        "schedule": {"n_steps": n_steps, "n_eval_batches": n_eval_batches,
                     "tokens_per_step": tokens_per_step, "batch_size": args.batch_size,
                     "seq_len": args.seq_len, "train_tokens": args.train_tokens},
        "probe_cfg": {"lr": args.lr, "m_projections": args.m_projections,
                      "lambda_kv": args.lambda_kv, "bridge_targets": "k+v",
                      "unfreeze": "lora", "trunk_policy": "frozen_base_lora_adapters",
                      "lora": spec.to_dict(), "n_lora_params_expected": n_lora_expected,
                      "amp_dtype": args.amp_dtype, "device": str(device)},
        "dataset": {"name": args.dataset_name, "config": args.dataset_config,
                    "eval_docs": args.eval_docs, "eval_offset_docs": args.eval_offset_docs},
        "tokenizer": {"name": args.tokenizer_name, "vocab_size": tokenizer.vocab_size},
        "code": code0, "recipe": recipe,
    }, indent=2, default=str))

    def _log_step(step: int, payload: object) -> None:
        if step % 50 == 0:
            print({"step": step, "loss": payload})

    merge_info: dict[str, Any] = {}
    partial_path = out_dir / "adapted_trunk.partial.pt"

    def _on_trained(probe) -> None:
        n_trainable = sum(p.numel() for p in probe.optimizer.param_groups[0]["params"])
        merge_info["n_lora_params"] = int(n_trainable)
        merge_info["n_lora_modules"] = len(lora_modules(probe.trunk))
        atomic_torch_save({"format": "lora_adapter_v1", "probe": PROBE_NAME,
                           "lora": spec.to_dict(), "state_dict": lora_state_dict(probe.trunk),
                           "steps": probe.global_step, "expected_steps": n_steps,
                           "source_checkpoint": str(args.checkpoint),
                           "source_checkpoint_sha256": source_sha256},
                          out_dir / "lora_adapter.pt")
        probe.trunk.eval()
        probe.head.eval()
        batch = next(iter(eval_factory()))
        x = batch["input_ids"].to(device)
        with torch.no_grad():
            before = probe.head(probe.trunk(x)).float()
            n_merged = merge_lora(probe.trunk)
            after = probe.head(probe.trunk(x)).float()
        merge_info["n_merged_modules"] = n_merged
        merge_info["merge_max_abs_logit_diff"] = float((before - after).abs().max())
        merge_info["merge_mean_abs_logit_diff"] = float((before - after).abs().mean())
        print(f"[lora_kv] trained {probe.global_step} steps; {n_trainable} adapter params in "
              f"{merge_info['n_lora_modules']} modules; merged {n_merged}; "
              f"max|Δlogit| {merge_info['merge_max_abs_logit_diff']:.3e}")
        atomic_torch_save(build_lora_probe_blob(
            probe.trunk, probe.head, trunk_kwargs=trainer.config.trunk_kwargs,
            tie_lm_head=trainer.config.tie_lm_head, source_checkpoint=str(args.checkpoint),
            probe_result={"status": ("trained_pre_eval" if probe.global_step == n_steps
                                     else "incomplete_training"),
                          "steps": probe.global_step, "expected_steps": n_steps, **merge_info},
            recipe=recipe, dev_mode=args.dev), partial_path)
        print(f"[lora_kv] pre-eval merged trunk snapshot -> {partial_path}")

    t0 = time.time()
    result = run_a1_probe(
        trunk, head, ckpt_loss, n_steps=n_steps, train_loader=train_factory(),
        eval_loader_factory=eval_factory, lambda_kv=args.lambda_kv,
        m_projections=args.m_projections, device=device, lr=args.lr, unfreeze="lora",
        lora_spec=spec, amp_dtype=amp_dtype, on_step=_log_step, on_trained=_on_trained,
        expected_steps=n_steps,
    )
    elapsed = time.time() - t0
    ok, why = code_identity_unchanged(_REPO_ROOT, code0)
    if not ok:
        raise SystemExit(f"[lora_kv] training code changed during the run ({why}) — "
                         "refusing the final save (partial snapshot retained)")
    payload = result.to_dict()
    payload["extra"] = {**payload.get("extra", {}), **merge_info,
                        "lora": spec.to_dict(), "trunk_params": n_base}
    payload["elapsed_seconds"] = elapsed
    payload["code"] = code0
    payload["dev_mode"] = bool(args.dev)
    atomic_torch_save(build_lora_probe_blob(
        trunk, head, trunk_kwargs=trainer.config.trunk_kwargs, tie_lm_head=trainer.config.tie_lm_head,
        source_checkpoint=str(args.checkpoint), probe_result=payload, recipe=recipe,
        dev_mode=args.dev), out_dir / "adapted_trunk.pt")
    partial_path.unlink(missing_ok=True)
    atomic_write_text(out_dir / "probe_result.json", json.dumps(payload, indent=2, default=str))
    print(f"\n[lora_kv] done  {result.gate_metric}={result.gate_value:.4f} "
          f"(threshold {result.gate_threshold}) passed={result.passed} "
          f"nll_delta={result.extra['nll_delta']:+.4f} in {elapsed:.1f}s -> "
          f"{out_dir / 'probe_result.json'}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
