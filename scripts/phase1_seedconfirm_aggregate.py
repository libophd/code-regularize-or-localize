from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

RUN_TAG_PATTERN = re.compile(r"^phase1_seedconfirm_s(\d+)$")
ARM_LABELS = {0: "baseline", 1: "sigreg_l0.01"}


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


def _condition_number(eval_metrics: dict[str, Any]) -> float:
    return float(eval_metrics["spectrum_max"]) / float(eval_metrics["spectrum_min"])


def _load_seed_runs(outputs_dir: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    run_dirs = sorted(
        d for d in outputs_dir.iterdir()
        if d.is_dir() and RUN_TAG_PATTERN.match(d.name)
    )
    if not run_dirs:
        raise SystemExit(
            f"[seedconfirm] no phase1_seedconfirm_s* run dirs under "
            f"{outputs_dir} — nothing to aggregate."
        )

    seeds: dict[int, dict[str, Any]] = {}
    missing: list[str] = []
    reference: tuple[str, dict[str, Any]] | None = None
    for run_dir in run_dirs:
        seed = int(RUN_TAG_PATTERN.match(run_dir.name).group(1))
        seeds[seed] = {}
        for arm_index, label in ARM_LABELS.items():
            arm_dir = run_dir / f"arm{arm_index}"
            config_path = arm_dir / "config.json"
            metrics_path = arm_dir / "sweep_metrics.json"
            if not config_path.is_file():
                missing.append(f"{run_dir.name}/arm{arm_index} (no config.json)")
                continue
            config = json.loads(config_path.read_text())

            config_seed = (config.get("trainer_config") or {}).get("seed")
            if config_seed != seed:
                raise SystemExit(
                    f"[seedconfirm] {config_path}: trainer_config.seed="
                    f"{config_seed} does not match run tag {run_dir.name} — "
                    f"a mis-tagged run must not enter the aggregate."
                )
            identity = _run_identity(config)
            if reference is None:
                reference = (f"{run_dir.name}/arm{arm_index}", identity)
            elif identity != reference[1]:
                raise SystemExit(
                    f"[seedconfirm] run-identity mismatch: "
                    f"{run_dir.name}/arm{arm_index} differs from "
                    f"{reference[0]} on dataset/tokenizer/schedule/sweep.\n"
                    f"  {reference[0]}: {reference[1]}\n"
                    f"  {run_dir.name}/arm{arm_index}: {identity}"
                )

            if not metrics_path.is_file():
                missing.append(
                    f"{run_dir.name}/arm{arm_index} ({label}: job not finished)"
                )
                continue
            payload = json.loads(metrics_path.read_text())
            results = payload.get("results") or []
            if len(results) != 1:
                raise SystemExit(
                    f"[seedconfirm] {metrics_path}: expected exactly 1 result "
                    f"(one arm per job), found {len(results)}."
                )
            result = results[0]
            if result.get("label") != label:
                raise SystemExit(
                    f"[seedconfirm] {metrics_path}: label "
                    f"{result.get('label')!r} != expected {label!r} for "
                    f"arm{arm_index}."
                )
            seeds[seed][label] = result["eval_metrics"]
    return seeds, missing


METRICS = [
    ("perplexity", "ppl", "{:.4f}"),
    ("mean_nll", "nll", "{:.4f}"),
    ("anisotropy_h", "aniso", "{:.5f}"),
    ("spectrum_top1_share", "top1_share", "{:.4f}"),
    (_condition_number, "cond#", "{:.1f}"),
]


def _metric_value(eval_metrics: dict[str, Any], key: Any) -> float:
    return key(eval_metrics) if callable(key) else float(eval_metrics[key])


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    return mean, std


def aggregate(seeds: dict[int, dict[str, Any]]) -> dict[str, Any]:
    per_arm: dict[str, Any] = {}
    for label in ARM_LABELS.values():
        rows = {s: m[label] for s, m in seeds.items() if label in m}
        if not rows:
            continue
        stats: dict[str, Any] = {"n": len(rows), "seeds": sorted(rows)}
        for key, name, _fmt in METRICS:
            values = [_metric_value(m, key) for m in rows.values()]
            mean, std = _mean_std(values)
            stats[name] = {"mean": mean, "std": std}
        per_arm[label] = stats

    deltas: dict[str, Any] = {"per_seed": {}, "summary": {}}
    both = {
        s: m for s, m in seeds.items()
        if "baseline" in m and "sigreg_l0.01" in m
    }
    for seed in sorted(both):
        base, sig = both[seed]["baseline"], both[seed]["sigreg_l0.01"]
        deltas["per_seed"][seed] = {
            "ppl_pct": 100.0 * (sig["perplexity"] / base["perplexity"] - 1.0),
            "nll_nats": sig["mean_nll"] - base["mean_nll"],
            "aniso_pct": 100.0 * (sig["anisotropy_h"] / base["anisotropy_h"] - 1.0),
            "top1_pct": 100.0
            * (sig["spectrum_top1_share"] / base["spectrum_top1_share"] - 1.0),
            "cond_delta": _condition_number(sig) - _condition_number(base),
        }
    if both:
        for field in next(iter(deltas["per_seed"].values())):
            values = [d[field] for d in deltas["per_seed"].values()]
            mean, std = _mean_std(values)
            deltas["summary"][field] = {"mean": mean, "std": std, "n": len(values)}
    return {"per_arm": per_arm, "within_seed_deltas": deltas}


def gate_comparison(
    gate_dir: Path,
    seeds: dict[int, dict[str, Any]],
    gate_seed: int,
    drift_flag_pct: float,
) -> dict[str, Any] | None:
    gate_path = gate_dir / "sweep_metrics.json"
    if not gate_path.is_file() or gate_seed not in seeds:
        return None
    gate_results = {
        r["label"]: r["eval_metrics"]
        for r in json.loads(gate_path.read_text()).get("results", [])
    }
    identity_note = None
    gate_config_path = gate_dir / "arm0" / "config.json"
    if gate_config_path.is_file():
        gate_config = json.loads(gate_config_path.read_text())
        gate_identity = _run_identity(gate_config)
        sample_config_path = next(
            (
                p / "arm0" / "config.json"
                for p in sorted(gate_dir.parent.iterdir())
                if RUN_TAG_PATTERN.match(p.name)
                and (p / "arm0" / "config.json").is_file()
            ),
            None,
        )
        if sample_config_path is not None:
            seedconf_identity = _run_identity(
                json.loads(sample_config_path.read_text())
            )
            mismatched = [
                block for block in ("dataset", "tokenizer", "schedule")
                if gate_identity[block] != seedconf_identity[block]
            ]
            if mismatched:
                identity_note = (
                    f"gate vs seedconfirm differ on {mismatched} — the "
                    f"reproduction check is NOT like-for-like."
                )

    comparison: dict[str, Any] = {
        "gate_seed": gate_seed,
        "identity_note": identity_note,
        "arms": {},
    }
    for label in ARM_LABELS.values():
        if label not in gate_results or label not in seeds[gate_seed]:
            continue
        gate_m, new_m = gate_results[label], seeds[gate_seed][label]
        arm: dict[str, Any] = {}
        for key, name, _fmt in METRICS:
            gate_v = _metric_value(gate_m, key)
            new_v = _metric_value(new_m, key)
            arm[name] = {
                "gate": gate_v,
                "seedconfirm": new_v,
                "drift_pct": 100.0 * (new_v / gate_v - 1.0),
            }
        arm["ppl_drift_flagged"] = (
            abs(arm["ppl"]["drift_pct"]) > drift_flag_pct
        )
        comparison["arms"][label] = arm
    return comparison


def print_report(
    report: dict[str, Any],
    missing: list[str],
    drift_flag_pct: float,
) -> None:
    if missing:
        print("!!! MISSING ARMS (aggregate is PARTIAL until these finish):")
        for item in missing:
            print(f"!!!   {item}")
        print()

    print("=== Per-arm across-seed stats ===")
    for label, stats in report["aggregate"]["per_arm"].items():
        print(f"{label}  (n={stats['n']}, seeds {stats['seeds']})")
        for _key, name, fmt in METRICS:
            mean, std = stats[name]["mean"], stats[name]["std"]
            std_txt = "±   n/a  " if std is None else f"± {std:<8.5f}"
            print(f"  {name:<10} {fmt.format(mean):>9} {std_txt}")

    deltas = report["aggregate"]["within_seed_deltas"]
    print("\n=== Within-seed deltas (sigreg_l0.01 − baseline) ===")
    header = f"{'seed':<6}{'Δppl%':>9}{'Δnll(nats)':>12}{'Δaniso%':>10}{'Δtop1%':>9}{'Δcond#':>9}"
    print(header)
    for seed, d in deltas["per_seed"].items():
        print(
            f"{seed:<6}{d['ppl_pct']:>+9.3f}{d['nll_nats']:>+12.4f}"
            f"{d['aniso_pct']:>+10.1f}{d['top1_pct']:>+9.1f}{d['cond_delta']:>+9.1f}"
        )
    if deltas["summary"]:
        s = deltas["summary"]
        n = s["ppl_pct"]["n"]

        def fmt_ms(field: str, fmt: str) -> str:
            std = s[field]["std"]
            return fmt.format(s[field]["mean"]) + (
                " ± n/a" if std is None else " ± " + fmt.format(std).lstrip("+")
            )

        print(
            f"mean over n={n} seeds: Δppl {fmt_ms('ppl_pct', '{:+.3f}')}%  "
            f"Δnll {fmt_ms('nll_nats', '{:+.4f}')} nats  "
            f"Δaniso {fmt_ms('aniso_pct', '{:+.1f}')}%  "
            f"Δcond# {fmt_ms('cond_delta', '{:+.1f}')}"
        )

    gate = report["gate_comparison"]
    if gate is None:
        print("\n(no gate comparison: gate results or gate-seed run absent)")
        return
    print(
        f"\n=== Seed-{gate['gate_seed']} reproduction check vs exit gate "
        f"(old stack -> new stack) ==="
    )
    if gate["identity_note"]:
        print(f"!!! {gate['identity_note']}")
    for label, arm in gate["arms"].items():
        flag = (
            f"  <-- STACK DRIFT (|Δppl| > {drift_flag_pct}%: absolute gate "
            f"numbers do not carry to the new stack; lean on within-seed deltas)"
            if arm["ppl_drift_flagged"]
            else ""
        )
        print(f"{label}{flag}")
        for _key, name, fmt in METRICS:
            m = arm[name]
            print(
                f"  {name:<10} gate {fmt.format(m['gate']):>9}  ->  "
                f"{fmt.format(m['seedconfirm']):>9}  ({m['drift_pct']:+.2f}%)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 1 seed-confirmation runs (n-seed mean±std "
        "+ within-seed free-lunch deltas + gate reproduction check)."
    )
    parser.add_argument(
        "--outputs",
        default="outputs",
        help="Directory holding phase1_seedconfirm_s*/ run dirs.",
    )
    parser.add_argument(
        "--gate-dir",
        default=None,
        help="Exit-gate run dir for the reproduction check "
        "(default: <outputs>/phase1_exitgate).",
    )
    parser.add_argument("--gate-seed", type=int, default=42)
    parser.add_argument(
        "--drift-flag-pct",
        type=float,
        default=0.5,
        help="Flag the reproduction check when |ppl drift| exceeds this %%.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Also write the aggregate as JSON to this path.",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs)
    if not outputs_dir.is_dir():
        raise SystemExit(f"[seedconfirm] outputs dir not found: {outputs_dir}")
    gate_dir = Path(args.gate_dir) if args.gate_dir else outputs_dir / "phase1_exitgate"

    seeds, missing = _load_seed_runs(outputs_dir)
    report = {
        "aggregate": aggregate(seeds),
        "gate_comparison": gate_comparison(
            gate_dir, seeds, args.gate_seed, args.drift_flag_pct
        ),
        "missing_arms": missing,
    }
    print_report(report, missing, args.drift_flag_pct)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n[seedconfirm] JSON written to {out_path}")


if __name__ == "__main__":
    main()
