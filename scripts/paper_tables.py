from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEEDS = ("42", "43", "44")


def _load(root: Path, rel: str):
    return json.loads((root / rel).read_text())


def table1_dose(root: Path) -> None:
    print("% --- Table 1: dose-response (tab:dose) ---")
    results = _load(root, "phase1_exitgate/sweep_metrics.json")["results"]
    by_label = {r["label"]: r["eval_metrics"] for r in results}
    base = by_label["baseline"]

    def row(label: str, m: dict) -> str:
        cond = m["spectrum_max"] / m["spectrum_min"]
        dppl = (
            "---" if label == "0 (baseline)"
            else f"${100 * (m['perplexity'] / base['perplexity'] - 1):+.2g}\\%$"
        )
        return (
            f"{label} & {m['perplexity']:.3f} & {dppl} & {m['anisotropy_h']:.4f}"
            f" & {m['spectrum_top1_share']:.4f} & {cond:.1f} \\\\"
        )

    print(row("0 (baseline)", base))
    for lam in ("0.001", "0.01", "0.1", "1.0"):
        print(row(lam, by_label[f"sigreg_l{lam}"]))


def table2_paired(root: Path) -> None:
    print("% --- Table 2: within-seed paired deltas (tab:paired) ---")
    agg = _load(root, "phase1_seedconfirm_aggregate.json")["aggregate"]["within_seed_deltas"]
    per, summ = agg["per_seed"], agg["summary"]
    rows = [
        ("anisotropy($h$)", "aniso_pct", "{:+.1f}\\%"),
        ("perplexity", "ppl_pct", "{:+.3f}\\%"),
        ("NLL (nats)", "nll_nats", "{:+.4f}"),
        ("cond\\#", "cond_delta", "{:+.1f}"),
        ("spectral top1", "top1_pct", "{:+.1f}\\%"),
    ]
    for name, key, fmt in rows:
        cells = " & ".join(fmt.format(per[s][key]) for s in SEEDS)
        s = summ[key]
        print(f"{name} & {cells} & {fmt.format(s['mean'])} $\\pm$ {fmt.format(s['std']).lstrip('+')} \\\\")


def _quant_cells(root: Path):
    cells: dict = defaultdict(dict)
    for path in sorted((root / "kv_quant").glob("*.json")):
        for r in json.loads(path.read_text()):
            if not isinstance(r, dict) or r.get("bits") is None:
                continue
            if r.get("targets") != "both":
                continue
            seed = next((s for s in SEEDS if f"s{s}" in r["model"]), None)
            kind = "ce_continue" if "ce_continue" in r["model"] else r["kind"]
            cells[(kind, seed)][r["config"]] = r["nll_delta_vs_ref"]
    return cells


def _mean(cells, kind: str, config: str) -> float:
    vals = [cells[(kind, s)][config] for s in SEEDS if config in cells.get((kind, s), {})]
    return sum(vals) / len(vals)


def table3_quant(root: Path) -> None:
    print("% --- Table 3: per-channel symmetric grid (tab:quant) ---")
    cells = _quant_cells(root)
    for bits in (8, 4, 3, 2):
        cfg = f"{bits}bit-per_channel-both"
        cols = [_mean(cells, k, cfg) for k in ("l_aux", "l_aux+l_sigreg", "a1_adapted")]
        ratios = [
            cells[("l_aux", s)][cfg] / cells[("a1_adapted", s)][cfg] for s in SEEDS
        ]
        ratio_txt = f"{min(ratios):.1f}--{max(ratios):.1f}$\\times$" if bits in (4, 3) else "---"
        print(
            f"{bits} & {cols[0]:+.3f} & {cols[1]:+.3f} & {cols[2]:+.3f} & {ratio_txt} \\\\"
        )


TABLE4_ROWS = [
    ("symmetric, per-token", "3bit-per_token-both"),
    ("symmetric, per-channel", "3bit-per_channel-both"),
    ("asymmetric, per-token", "3bit-per_token-both-asym"),
    ("asymmetric, per-channel", "3bit-per_channel-both-asym"),
    ("mean-sub $+$ sym., per-token", "3bit-per_token-both-meansub"),
    ("mean-sub $+$ sym., per-channel", "3bit-per_channel-both-meansub"),
]

TABLE5_ROWS = [
    ("mixed (pc-K/pt-V), symmetric, group-free", "3bit-mixed-both"),
    ("symmetric, per-token, per-head ($g{=}64$)", "3bit-per_token-both-g64"),
    ("asymmetric, per-token, per-head ($g{=}64$)", "3bit-per_token-both-asym-g64"),
    ("symmetric, per-token, $g{=}32$", "3bit-per_token-both-g32"),
    ("symmetric, per-channel, $g{=}32$", "3bit-per_channel-both-g32"),
    ("mixed (pc-K/pt-V), symmetric, $g{=}32$", "3bit-mixed-both-g32"),
    ("asymmetric, per-token, $g{=}32$", "3bit-per_token-both-asym-g32"),
    ("KIVI-style: mixed, asym., $g{=}32$", "3bit-mixed-both-asym-g32"),
]

ARMS = ("l_aux", "l_aux+l_sigreg", "a1_adapted", "ce_continue")


def _arm_table(root: Path, rows, header: str) -> None:
    print(header)
    cells = _quant_cells(root)
    for label, cfg in rows:
        cols = " & ".join(f"${_mean(cells, arm, cfg):+.3f}$" for arm in ARMS)
        print(f"{label} & {cols} \\\\")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the paper's table bodies from artifacts.")
    parser.add_argument("--artifacts", action="store_true", help="read paper/artifacts/ instead of outputs/")
    args = parser.parse_args()
    root = REPO / ("paper/artifacts" if args.artifacts else "outputs")
    table1_dose(root)
    print()
    table2_paired(root)
    print()
    table3_quant(root)
    print()
    _arm_table(root, TABLE4_ROWS, "% --- Table 4: 3-bit controls (tab:controls) ---")
    print()
    _arm_table(root, TABLE5_ROWS, "% --- Table 5: KIVI-style grid (tab:kivi) ---")


if __name__ == "__main__":
    main()
