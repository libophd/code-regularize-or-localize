from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch

from jepa_lm.revision.envcheck import (
    ANCHOR_FREEZE_REL_PATH,
    ANCHOR_FREEZE_SHA256,
    check_freeze_complete,
    run_nvidia_smi,
)
from jepa_lm.revision.guards import (
    enforce_single_visible_gpu,
    publish_no_replace_isolated,
)
from jepa_lm.revision.hf_pins import (
    PINNED_HF_REVISIONS,
    hf_cache_dir,
    resolve_cached_revision,
)
from jepa_lm.revision.sweep_configs import (
    CANONICAL_PILOT_LAUNCH,
    CANONICAL_PILOT_MANIFEST_OUT,
    CANONICAL_RECEIPT_PATH,
    REGISTERED_ARMS,
    REGISTERED_MODEL_SHA256,
    REGISTERED_SWEEP_OUT_ROOT,
    REVISION_OUT_ROOT,
    SWEEPS,
    assert_isolated_out_path,
    assert_legacy_trees_locked,
    expected_sweep_argv_tail,
)

def _fail(msg: str) -> None:
    raise SystemExit(f"[preflight] FAIL: {msg}")


def _digest(path: Path) -> dict[str, str | int]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            sha.update(chunk)
            md5.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def main() -> None:
    p = argparse.ArgumentParser(description="Registered-launch preflight + manifest.")
    p.add_argument("--model", required=True)
    p.add_argument("--arm", required=True, choices=list(REGISTERED_ARMS))
    p.add_argument("--source-seed", type=int, required=True)
    p.add_argument("--continuation-seed", type=int, default=None)
    p.add_argument("--sweep", required=True, choices=sorted(SWEEPS))
    p.add_argument("--out-json", required=True)
    p.add_argument("--manifest-out", required=True)
    p.add_argument("--allow-dirty", action="store_true",
                   help="dev only: registered launches require a clean checkout")
    p.add_argument("--crosscheck-receipt", default=None,
                   help="pilot comparability receipt — REQUIRED for every "
                        "retained launch except the explicit pilot bootstrap")
    p.add_argument("--pilot-bootstrap", action="store_true",
                   help="narrow exemption for THE pilot itself (l_aux/s42/st3 "
                        "only): launch without a crosscheck receipt")
    p.add_argument("--launch", action="store_true",
                   help="immediately exec the authorized sweep after writing the "
                        "manifest (one command, no bypass window)")
    args = p.parse_args()

    manifest_out = Path(args.manifest_out).resolve()
    lock_path = Path(str(Path(args.out_json).resolve()) + ".lock")
    for other in (Path(args.model).resolve(), Path(args.out_json).resolve()):
        if manifest_out == other:
            _fail(f"--manifest-out overlaps another launch path: {other}")
    if manifest_out == lock_path or lock_path in manifest_out.parents:
        _fail(f"--manifest-out is at/inside the sweep lock path {lock_path}")
    out_resolved = Path(args.out_json).resolve()
    if manifest_out in out_resolved.parents:
        _fail(f"--manifest-out {manifest_out} is an ancestor of the output path — "
              "the child could never create its output")
    if out_resolved == manifest_out or out_resolved in manifest_out.parents:
        _fail(f"--manifest-out {manifest_out} is at/below the output path — "
              "writing it would occupy the output location and self-block the launch")
    assert_isolated_out_path(
        args.out_json, _REPO_ROOT, kind="sweep out-json",
        required_root=None if args.allow_dirty else REGISTERED_SWEEP_OUT_ROOT,
        forbidden_root=REGISTERED_SWEEP_OUT_ROOT if args.allow_dirty else None,
        allow_canonical_slot="pilot-result" if args.pilot_bootstrap else None)
    assert_isolated_out_path(
        args.manifest_out, _REPO_ROOT, kind="manifest-out",
        required_root=None if args.allow_dirty else REVISION_OUT_ROOT,
        allow_canonical_slot="pilot-manifest" if args.pilot_bootstrap else None)
    if not args.allow_dirty:
        n_locked = assert_legacy_trees_locked(_REPO_ROOT)
        print(f"[preflight] frozen legacy trees verified read-only "
              f"({n_locked} entries)")

    if not torch.cuda.is_available():
        _fail("CUDA unavailable in this environment")
    enforce_single_visible_gpu(torch.device("cuda"))
    nvidia_smi = run_nvidia_smi()

    packages = check_freeze_complete(_REPO_ROOT, rel_path=ANCHOR_FREEZE_REL_PATH,
                                     expected_sha256=ANCHOR_FREEZE_SHA256)

    revisions = {
        repo: resolve_cached_revision(kind, repo)
        for kind, repo in (("models", "mistralai/Mistral-7B-v0.1"),
                           ("datasets", "HuggingFaceFW/fineweb"))
    }

    def _git(cmd: list[str]) -> str:
        return subprocess.check_output(["git", *cmd], cwd=_REPO_ROOT, text=True).strip()

    sha = _git(["rev-parse", "HEAD"])
    dirty = bool(_git(["status", "--porcelain"]))
    if dirty and not args.allow_dirty:
        _fail("checkout is dirty — registered launches run from a clean checkout "
              "(commit/stash, or pass --allow-dirty for a disposable run)")

    model_path = Path(args.model)
    if not model_path.is_file():
        _fail(f"model not found: {model_path}")
    model = _digest(model_path)
    entry = REGISTERED_MODEL_SHA256.get(str(model["sha256"]))
    if entry != (args.arm, args.source_seed):
        _fail(f"model sha256 {str(model['sha256'])[:16]}… is registered as {entry}, "
              f"claimed ({args.arm!r}, {args.source_seed})")

    receipt_bytes = None
    if args.allow_dirty:
        if args.crosscheck_receipt is not None or args.pilot_bootstrap:
            _fail("--allow-dirty is dev-only and takes neither "
                  "--crosscheck-receipt nor --pilot-bootstrap")
        mode = "dev"
    elif args.pilot_bootstrap:
        if args.crosscheck_receipt is not None:
            _fail("--pilot-bootstrap and --crosscheck-receipt are mutually "
                  "exclusive — the bootstrap is the launch that MINTS the receipt")
        got = {"model": args.model, "arm": args.arm,
               "source_seed": args.source_seed,
               "continuation_seed": args.continuation_seed,
               "sweep": args.sweep, "out_json": args.out_json}
        if got != CANONICAL_PILOT_LAUNCH \
                or args.manifest_out != CANONICAL_PILOT_MANIFEST_OUT:
            _fail("--pilot-bootstrap authorizes ONLY the canonical pilot "
                  f"artifact: launch {CANONICAL_PILOT_LAUNCH} with "
                  f"--manifest-out {CANONICAL_PILOT_MANIFEST_OUT} (got {got} "
                  f"with --manifest-out {args.manifest_out!r}); every other "
                  "launch needs --crosscheck-receipt")
        mode = "pilot-bootstrap"
        print("[preflight] pilot bootstrap: THE canonical pilot, launching "
              "WITHOUT a receipt (single-shot pinned paths)")
    elif args.crosscheck_receipt is not None:
        if (args.arm, args.source_seed, args.sweep) == (
                CANONICAL_PILOT_LAUNCH["arm"],
                CANONICAL_PILOT_LAUNCH["source_seed"],
                CANONICAL_PILOT_LAUNCH["sweep"]):
            _fail("the canonical pilot identity (l_aux, 42, st3) launches "
                  "ONLY via --pilot-bootstrap — a retained receipt-mode rerun "
                  "would shadow the certified pilot artifact and dodge the "
                  "merge's byte binding")
        if args.crosscheck_receipt != CANONICAL_RECEIPT_PATH:
            _fail(f"--crosscheck-receipt {args.crosscheck_receipt!r} != the "
                  f"canonical receipt {CANONICAL_RECEIPT_PATH!r} — the wrapper "
                  "path is pinned (single receipt, single location)")
        sys.path.insert(0, str(_REPO_ROOT / "scripts" / "revision"))
        from pilot_crosscheck import verify_receipt
        receipt_bytes = Path(args.crosscheck_receipt).read_bytes()
        from jepa_lm.revision.sweep_configs import strict_json_loads
        receipt = strict_json_loads(receipt_bytes.decode())
        try:
            verify_receipt(receipt, _REPO_ROOT)
        except AssertionError as e:
            _fail(f"crosscheck receipt invalid: {e}")
        mode = "receipt"
        print(f"[preflight] crosscheck receipt verified "
              f"(pilot {receipt['pilot']['sha256'][:12]}…, PASS)")
    else:
        _fail("--crosscheck-receipt is REQUIRED for retained launches "
              "(--pilot-bootstrap only for the canonical pilot; --allow-dirty "
              "runs are dev-only and never merge)")

    out_json = Path(args.out_json)
    if out_json.exists():
        _fail(f"output already exists: {out_json}")
    if Path(str(out_json) + ".lock").exists():
        _fail(f"lock already exists: {out_json}.lock")

    manifest = {
        "schema_version": "launch-manifest-v1",
        "mode": mode,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {"sha": sha, "dirty": dirty},
        "launch": {"model": str(model_path), "arm": args.arm,
                   "source_seed": args.source_seed,
                   "continuation_seed": args.continuation_seed,
                   "sweep": args.sweep, "out_json": str(out_json)},
        "packages": packages,
        "python": sys.version.split()[0],
        "cuda": {"torch_cuda": torch.version.cuda,
                 "device_name": torch.cuda.get_device_name(0),
                 "visible_devices": torch.cuda.device_count(),
                 "nvidia_smi": nvidia_smi},
        "hf_cache_dir": str(hf_cache_dir()),
        "hf_revisions": revisions,
        "hf_pinned": PINNED_HF_REVISIONS,
        "models": [model],
        "crosscheck_receipt": (
            None if receipt_bytes is None else {
                "path": args.crosscheck_receipt,
                "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "content": receipt_bytes.decode(),
            }),
        "command": sys.argv,
    }
    assert_isolated_out_path(
        args.manifest_out, _REPO_ROOT, kind="manifest-out",
        required_root=None if args.allow_dirty else REVISION_OUT_ROOT,
        allow_canonical_slot="pilot-manifest" if args.pilot_bootstrap else None)
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    publish_no_replace_isolated(_REPO_ROOT, manifest_out, manifest_text)
    manifest_sha = hashlib.sha256(manifest_text.encode()).hexdigest()
    print(json.dumps(manifest, indent=2))
    print(f"[preflight] PASS — manifest -> {manifest_out} (sha256 {manifest_sha[:16]}…)")
    if not args.launch:
        print(f"[preflight] launch with: --manifest {manifest_out} "
              f"--manifest-sha256 {manifest_sha}")
        return
    cmd = [sys.executable,
           str(_REPO_ROOT / "scripts" / "revision" / "kv_quant_sweep.py"),
           *expected_sweep_argv_tail(
               model=str(model_path), arm=args.arm,
               source_seed=args.source_seed,
               continuation_seed=args.continuation_seed,
               sweep=args.sweep, out_json=str(out_json),
               manifest=args.manifest_out, manifest_sha256=manifest_sha)]
    if args.allow_dirty:
        cmd += ["--dev-dirty-ok"]
    print(f"[preflight] exec: {' '.join(cmd)}")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
