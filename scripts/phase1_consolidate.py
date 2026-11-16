from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _arm_sort_key(result: dict[str, Any]) -> tuple[int, float]:
    weight = result.get("sigreg_weight")
    return (0, 0.0) if weight is None else (1, float(weight))


def _run_identity(config: dict[str, Any]) -> dict[str, Any]:
    sweep = {
        k: v for k, v in (config.get("sweep") or {}).items() if k != "arm_index"
    }
    return {
        "dataset": config.get("dataset"),
        "tokenizer": config.get("tokenizer"),
        "schedule": config.get("schedule"),
        "sweep": sweep,
    }


def _load_arms(run_dir: Path) -> list[dict[str, Any]]:
    config_files = sorted(run_dir.glob("arm*/config.json"))
    if not config_files:
        raise SystemExit(
            f"[consolidate] no arm*/config.json under {run_dir} — nothing to "
            f"consolidate (expected per-arm subdirs arm0/, arm1/, ...)."
        )

    arms: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for config_path in config_files:
        arm_dir = config_path.parent
        config = json.loads(config_path.read_text())
        arm_index = (config.get("sweep") or {}).get("arm_index")
        if arm_index is None:
            raise SystemExit(
                f"[consolidate] {config_path} has no per-arm arm_index "
                f"(sweep.arm_index is null) — that is a whole-sweep config, "
                f"not a one-arm-per-job run. Point this script at a directory "
                f"of --arm-index jobs."
            )
        metrics_path = arm_dir / "sweep_metrics.json"
        if not metrics_path.is_file():
            incomplete.append(f"{arm_dir.name} (arm_index={arm_index})")
            continue
        arms.append({
            "dir": arm_dir,
            "arm_index": int(arm_index),
            "identity": _run_identity(config),
            "payload": json.loads(metrics_path.read_text()),
        })

    if incomplete:
        raise SystemExit(
            "[consolidate] these arm dirs have config.json but no "
            "sweep_metrics.json — their jobs did not finish:\n  "
            + "\n  ".join(incomplete)
            + "\nRe-run the missing arms before consolidating."
        )
    return arms


def _validate(arms: list[dict[str, Any]]) -> None:
    reference = arms[0]
    for arm in arms[1:]:
        if arm["identity"] != reference["identity"]:
            raise SystemExit(
                f"[consolidate] run-identity mismatch: {arm['dir'].name} "
                f"differs from {reference['dir'].name} on "
                f"dataset/tokenizer/schedule/sweep. A stale arm dir from a "
                f"different run must not be consolidated into one comparison.\n"
                f"  {reference['dir'].name}: {reference['identity']}\n"
                f"  {arm['dir'].name}: {arm['identity']}"
            )

    lambdas = list((reference["identity"]["sweep"] or {}).get("lambdas") or [])
    indices = [arm["arm_index"] for arm in arms]
    if len(indices) != len(set(indices)):
        dups = sorted({i for i in indices if indices.count(i) > 1})
        raise SystemExit(
            f"[consolidate] duplicate arm_index {dups} — each arm index must "
            f"appear exactly once. Check for stale arm dirs."
        )
    expected = set(range(len(lambdas) + 1))
    actual = set(indices)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise SystemExit(
            f"[consolidate] incomplete sweep: arm indices {sorted(actual)} != "
            f"expected {sorted(expected)} (0 = baseline, 1..{len(lambdas)} = "
            f"the {len(lambdas)} λ values)"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {unexpected}" if unexpected else "")
            + "."
        )

    expected_labels = {"baseline"} | {f"sigreg_l{w}" for w in lambdas}
    actual_labels: set[str] = set()
    for arm in arms:
        for result in arm["payload"].get("results", []):
            actual_labels.add(str(result.get("label", "?")))
    if actual_labels != expected_labels:
        raise SystemExit(
            f"[consolidate] arm-label set {sorted(actual_labels)} != expected "
            f"{sorted(expected_labels)}."
        )


def consolidate(run_dir: Path, out_name: str) -> tuple[Path, list[dict[str, Any]]]:
    arms = _load_arms(run_dir)
    _validate(arms)

    results: list[dict[str, Any]] = []
    elapsed_total = 0.0
    for arm in arms:
        payload = arm["payload"]
        elapsed_total += float(payload.get("elapsed_seconds") or 0.0)
        results.extend(payload.get("results", []))
    results.sort(key=_arm_sort_key)

    out_path = run_dir / out_name
    out_path.write_text(json.dumps(
        {
            "elapsed_seconds": elapsed_total,
            "results": results,
            "consolidated_from": [
                str(arm["dir"] / "sweep_metrics.json") for arm in arms
            ],
        },
        indent=2,
        default=str,
    ))
    return out_path, results


def print_table(results: list[dict[str, Any]]) -> None:
    headers = ["label", "λ", "ppl", "aniso", "top1_share", "steps"]
    rows: list[list[str]] = []
    for r in results:
        m = r["eval_metrics"]
        weight = r.get("sigreg_weight")
        rows.append([
            str(r.get("label", "?")),
            "—" if weight is None else f"{float(weight):g}",
            f"{m['perplexity']:.4f}",
            f"{m['anisotropy_h']:+.4f}",
            f"{m['spectrum_top1_share']:.4f}",
            str(r.get("n_steps_trained", "?")),
        ])
    widths = [
        max(len(h), *(len(row[i]) for row in rows))
        for i, h in enumerate(headers)
    ]
    print("=== Consolidated Phase 1 sweep ===")
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and consolidate per-arm Phase 1 sweep results."
    )
    parser.add_argument(
        "run_dir",
        help="Directory holding the per-arm arm*/ subdirs.",
    )
    parser.add_argument(
        "--out-name",
        default="sweep_metrics.json",
        help="Name of the consolidated file written into run_dir.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"[consolidate] run_dir not found: {run_dir}")

    out_path, results = consolidate(run_dir, args.out_name)
    print_table(results)
    print(f"\n[consolidate] {len(results)} arms -> {out_path}")


if __name__ == "__main__":
    main()
