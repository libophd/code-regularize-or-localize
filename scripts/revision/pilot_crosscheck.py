from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from jepa_lm.revision.envcheck import (
    ANCHOR_FREEZE_REL_PATH,
    ANCHOR_FREEZE_SHA256,
    load_verified_freeze,
)

from jepa_lm.revision.guards import publish_no_replace_isolated
from jepa_lm.revision.sweep_configs import (
    ANISO_REPLAY_ABS_TOL,
    CANONICAL_ANCHOR_PATH,
    CANONICAL_RECEIPT_PATH,
    METRIC_TRIPLE,
    OVERLAP_CONFIGS,
    RECEIPT_SCHEMA_VERSION,
    REGISTERED_MODEL_SHA256,
    strict_json_loads,
    validate_receipt_content,
    validate_split_exact,
    validate_split_payloads,
)

DEFAULT_PUBLISHED = CANONICAL_ANCHOR_PATH
RECEIPT_SCHEMA = RECEIPT_SCHEMA_VERSION
def _anchor_freeze(repo_root) -> dict[str, str]:
    return load_verified_freeze(repo_root, rel_path=ANCHOR_FREEZE_REL_PATH,
                                expected_sha256=ANCHOR_FREEZE_SHA256)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_published_input(repo_root: Path, published_path: Path) -> str:
    manifest = json.loads((repo_root / "paper" / "artifacts" / "manifest.json").read_text())
    try:
        rel = published_path.resolve().relative_to(
            (repo_root / "paper" / "artifacts").resolve())
    except ValueError:
        raise AssertionError(
            f"published input {published_path} is not inside the tracked "
            "paper/artifacts bundle — the gitignored outputs/ copy is mutable "
            "and never the authority"
        ) from None
    entry = next((e for e in manifest["files"] if e["path"] == str(rel)), None)
    if entry is None:
        raise AssertionError(f"{rel} is not recorded in the artifact manifest")
    actual = _sha256_file(published_path)
    if actual != entry["sha256"]:
        raise AssertionError(
            f"published artifact digest {actual[:16]}… != manifest-recorded "
            f"{entry['sha256'][:16]}… — the tracked bundle was modified"
        )
    return actual


def run_crosscheck(
    pilot_payload: Mapping[str, Any],
    published_rows: list[dict],
    *,
    kind: str,
    arm: str,
    source_seed: int,
    freeze_pins: Mapping[str, str],
    model_registry: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, dict]:
    validate_split_payloads({"st3": dict(pilot_payload)}, freeze_pins=freeze_pins)
    rows = pilot_payload["rows"]
    validate_split_exact("st3", rows)
    r0 = rows[0]
    if (r0.get("arm"), r0.get("source_seed")) != (arm, source_seed):
        raise AssertionError(
            f"pilot rows claim ({r0.get('arm')!r}, {r0.get('source_seed')!r}), "
            f"expected ({arm!r}, {source_seed})"
        )
    registry = REGISTERED_MODEL_SHA256 if model_registry is None else model_registry
    msha = (pilot_payload.get("env") or {}).get("model_sha256")
    if registry.get(msha) != (arm, source_seed):
        raise AssertionError(
            f"pilot model_sha256 {str(msha)[:16]}… is registered as "
            f"{registry.get(msha)}, expected ({arm!r}, {source_seed})"
        )
    pub = {r["config"]: r for r in published_rows
           if r.get("kind") == kind and r.get("config") in OVERLAP_CONFIGS}
    pil = {r["config"]: r for r in rows if r.get("config") in OVERLAP_CONFIGS}
    missing = [c for c in OVERLAP_CONFIGS if c not in pub or c not in pil]
    if missing:
        raise AssertionError(f"overlap cells absent (kind={kind!r}): {missing}")
    report: dict[str, dict] = {}
    mismatches: dict[str, dict] = {}
    for cfg in OVERLAP_CONFIGS:
        entry = {"published": {k: pub[cfg][k] for k in METRIC_TRIPLE},
                 "pilot": {k: pil[cfg][k] for k in METRIC_TRIPLE}}
        report[cfg] = entry
        exact_bad = any(entry["published"][k] != entry["pilot"][k]
                        for k in ("mean_nll", "perplexity"))
        aniso_bad = abs(entry["published"]["anisotropy_h"]
                        - entry["pilot"]["anisotropy_h"]) > ANISO_REPLAY_ABS_TOL
        if exact_bad or aniso_bad:
            mismatches[cfg] = entry
    if mismatches:
        raise AssertionError(
            "COMPARABILITY BREAK — pilot does not reproduce the published "
            f"cells (NLL/ppl exact; anisotropy within {ANISO_REPLAY_ABS_TOL}): "
            f"{mismatches}. Resolve before ANY pass-1 data is retained."
        )
    return report


def assert_pilot_commit_binding(
    pilot_payload: Mapping[str, Any], head: str,
) -> None:
    pgit = pilot_payload.get("git") or {}
    if pgit.get("sha") != head or pgit.get("dirty") is not False:
        raise AssertionError(
            f"pilot payload git {pgit!r} != the receipt's clean commit "
            f"{head!r} — an old-code pilot cannot be re-minted under a new "
            "commit; rerun the pilot at the current HEAD"
        )


def verify_receipt(receipt: Mapping[str, Any], repo_root: Path) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise AssertionError(f"receipt schema {receipt.get('schema_version')!r}")
    if receipt.get("verdict") != "PASS":
        raise AssertionError(f"receipt verdict {receipt.get('verdict')!r} != PASS")
    if not receipt.get("timestamp_utc") or not receipt.get("command"):
        raise AssertionError("receipt missing timestamp/command evidence")
    pub = receipt.get("published") or {}
    if pub.get("path") != DEFAULT_PUBLISHED:
        raise AssertionError(
            f"receipt's published path {pub.get('path')!r} is not the "
            f"canonical anchor {DEFAULT_PUBLISHED!r}"
        )
    recorded = verify_published_input(repo_root, repo_root / DEFAULT_PUBLISHED)
    if recorded != pub.get("sha256"):
        raise AssertionError("receipt's published digest no longer matches the bundle")
    pilot = receipt.get("pilot") or {}
    ppath_str = str(pilot.get("path", ""))
    if Path(ppath_str).is_absolute() or ".." in Path(ppath_str).parts:
        raise AssertionError(
            f"receipt pilot path {ppath_str!r} must be repo-relative")
    ppath = repo_root / ppath_str
    try:
        pilot_bytes = ppath.read_bytes()
    except OSError:
        raise AssertionError(f"receipt's pilot file {ppath_str!r} is unreadable") from None
    if hashlib.sha256(pilot_bytes).hexdigest() != pilot.get("sha256"):
        raise AssertionError("receipt's pilot digest does not match the file on disk")
    git = receipt.get("git") or {}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root,
                                   text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                         cwd=repo_root, text=True).strip())
    if git.get("sha") != head or git.get("dirty") is not False or dirty:
        raise AssertionError(
            f"receipt git state {git} does not match the current clean HEAD {head}"
        )
    validate_receipt_content(receipt, expect_git_sha=head)
    pilot_payload = strict_json_loads(pilot_bytes.decode())
    assert_pilot_commit_binding(pilot_payload, head)
    published_rows = json.loads((repo_root / DEFAULT_PUBLISHED).read_text())
    report = run_crosscheck(
        pilot_payload, published_rows,
        kind=str(receipt.get("kind")), arm=str(receipt.get("arm")),
        source_seed=receipt.get("source_seed"),
        freeze_pins=_anchor_freeze(repo_root),
    )
    if receipt.get("cells") != report:
        raise AssertionError(
            "receipt cells do not match the recomputed crosscheck report"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Authenticated pilot comparability gate.")
    p.add_argument("--pilot", required=True, help="pilot st3 result JSON")
    p.add_argument("--published", default=str(_REPO_ROOT / DEFAULT_PUBLISHED),
                   help="TRACKED artifact copy (digest-verified vs the manifest)")
    p.add_argument("--kind", default="l_aux")
    p.add_argument("--arm", required=True)
    p.add_argument("--source-seed", type=int, required=True)
    p.add_argument("--receipt-out", required=True)
    args = p.parse_args()

    if args.receipt_out != CANONICAL_RECEIPT_PATH:
        raise SystemExit(
            f"[crosscheck] FAIL: --receipt-out {args.receipt_out!r} != the "
            f"canonical receipt path {CANONICAL_RECEIPT_PATH!r} — one receipt, "
            "one location (preflight and the merger accept nothing else)")
    published_path = Path(args.published)
    published_sha = verify_published_input(_REPO_ROOT, published_path)
    pilot_path = Path(args.pilot)
    pilot_bytes = pilot_path.read_bytes()
    pilot_sha = hashlib.sha256(pilot_bytes).hexdigest()
    pilot_payload = strict_json_loads(pilot_bytes.decode())
    published_rows = json.loads(published_path.read_text())

    def _git(cmd: list[str]) -> str:
        return subprocess.check_output(["git", *cmd], cwd=_REPO_ROOT, text=True).strip()

    head = _git(["rev-parse", "HEAD"])
    dirty = bool(_git(["status", "--porcelain"]))
    if dirty:
        raise SystemExit(
            "[crosscheck] FAIL: checkout is dirty — receipts are minted only "
            "from the clean commit the pilot ran at")
    assert_pilot_commit_binding(pilot_payload, head)

    report = run_crosscheck(
        pilot_payload, published_rows, kind=args.kind, arm=args.arm,
        source_seed=args.source_seed,
        freeze_pins=_anchor_freeze(_REPO_ROOT),
    )

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {"sha": head, "dirty": dirty},
        "pilot": {"path": str(pilot_path.resolve().relative_to(_REPO_ROOT.resolve())),
                  "sha256": pilot_sha,
                  "content": pilot_bytes.decode()},
        "published": {"path": str(published_path.resolve().relative_to(_REPO_ROOT.resolve())),
                      "sha256": published_sha},
        "kind": args.kind, "arm": args.arm, "source_seed": args.source_seed,
        "cells": report,
        "verdict": "PASS",
        "command": sys.argv,
    }
    validate_receipt_content(receipt, expect_git_sha=head)
    out = Path(args.receipt_out)
    from jepa_lm.revision.sweep_configs import (
        REVISION_OUT_ROOT,
        assert_isolated_out_path,
    )
    assert_isolated_out_path(out, _REPO_ROOT, kind="receipt-out",
                             required_root=REVISION_OUT_ROOT,
                             allow_canonical_slot="receipt")
    publish_no_replace_isolated(_REPO_ROOT, out, json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    for cfg, entry in report.items():
        print(f"[crosscheck] {cfg:>24}: triple reproduced exactly ✓")
    print(f"[crosscheck] PASS — receipt -> {out}")


if __name__ == "__main__":
    main()
