from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from jepa_lm.revision.envcheck import (
    ANCHOR_FREEZE_REL_PATH,
    ANCHOR_FREEZE_SHA256,
    load_verified_freeze,
)
from jepa_lm.revision.guards import publish_no_replace_isolated
from jepa_lm.revision.sweep_configs import (
    REVISION_OUT_ROOT,
    assert_isolated_out_path,
    merge_sweep_outputs,
    strict_json_loads,
    validate_merge_identity,
    validate_merged_envs,
    validate_split_payloads,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Merge one model's three sweep splits.")
    p.add_argument("--st3", required=True)
    p.add_argument("--bt3", required=True)
    p.add_argument("--bt4", required=True)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    assert_isolated_out_path(args.out_json, _REPO_ROOT, kind="merge out-json",
                             required_root=REVISION_OUT_ROOT)

    import hashlib

    raw = {name: Path(getattr(args, name)).read_bytes()
           for name in ("st3", "bt3", "bt4")}
    payloads = {name: strict_json_loads(b.decode()) for name, b in raw.items()}
    split_digests = {name: hashlib.sha256(b).hexdigest() for name, b in raw.items()}
    validate_split_payloads(
        payloads,
        freeze_pins=load_verified_freeze(
            _REPO_ROOT, rel_path=ANCHOR_FREEZE_REL_PATH,
            expected_sha256=ANCHOR_FREEZE_SHA256),
        split_digests=split_digests)
    rows_by_sweep = {name: p["rows"] for name, p in payloads.items()}
    merged = merge_sweep_outputs(rows_by_sweep)
    merged["env"] = validate_merged_envs(
        {name: p.get("env") for name, p in payloads.items()})
    validate_merge_identity(merged["env"], merged["rows"])
    merged["schema_version"] = "groupsweep-merged-v1"
    merged["sources"] = {name: {"git": p.get("git"), "command": p.get("command"),
                                "dev_mode": p.get("dev_mode"),
                                "input_sha256": split_digests[name],
                                "launch_manifest": p.get("launch_manifest"),
                                "launch_manifest_sha256": p.get("launch_manifest_sha256"),
                                "launch_manifest_json": p.get("launch_manifest_json")}
                         for name, p in payloads.items()}
    out = Path(args.out_json)
    assert_isolated_out_path(out, _REPO_ROOT, kind="merge out-json",
                             required_root=REVISION_OUT_ROOT)
    publish_no_replace_isolated(_REPO_ROOT, out, json.dumps(merged, indent=2, allow_nan=False))
    print(f"[merge] reference_nll={merged['reference_nll']:.6f}, "
          f"{len(merged['rows'])} cells -> {out}")


if __name__ == "__main__":
    main()
