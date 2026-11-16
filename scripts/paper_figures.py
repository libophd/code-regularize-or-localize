from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "figures"
DATA = REPO / "outputs"

COLORS = {"baseline": "#0072B2", "sigreg": "#E69F00", "a1": "#009E73"}
MARKERS = {"baseline": "o", "sigreg": "s", "a1": "^"}
LABELS = {"baseline": "baseline", "sigreg": "SIGReg($h$)", "a1": "A1-adapted"}
KIND_MAP = {"l_aux": "baseline", "l_aux+l_sigreg": "sigreg", "a1_adapted": "a1"}
SEEDS = ("42", "43", "44")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
})


def load_quant() -> dict:
    data: dict = {k: {s: {} for s in SEEDS} for k in COLORS}
    for seed in SEEDS:
        for part in ("phase1", "a1"):
            path = DATA / "kv_quant" / f"results_s{seed}_{part}.json"
            for row in json.loads(path.read_text()):
                if row["bits"] is None or row["targets"] != "both":
                    continue
                key = KIND_MAP[row["kind"]]
                data[key][seed][(row["bits"], row["granularity"])] = row["nll_delta_vs_ref"]
    return data


def fig_quant() -> None:
    data = load_quant()
    bits = [2, 3, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, gran, title in zip(
        axes, ("per_token", "per_channel"), ("per-token scaling", "per-channel scaling")
    ):
        for key in ("baseline", "sigreg", "a1"):
            means = []
            for b in bits:
                vals = [data[key][s][(b, gran)] for s in SEEDS]
                means.append(sum(vals) / len(vals))
                ax.plot(
                    [b] * len(vals), vals,
                    linestyle="none", marker=MARKERS[key], color=COLORS[key],
                    markersize=3, alpha=0.45,
                )
            ax.plot(
                bits, means, marker=MARKERS[key], color=COLORS[key],
                label=LABELS[key],
            )
        ax.set_yscale("log")
        ax.set_xticks(bits)
        ax.set_xlabel("bits")
        ax.set_title(title)
        ax.invert_xaxis()
    axes[0].set_ylabel(r"$\Delta$NLL vs own fp32 ref (nats)")
    axes[0].legend(frameon=False, loc="upper left")
    base_4pc = sum(data["baseline"][s][(4, "per_channel")] for s in SEEDS) / 3
    a1_3pc = sum(data["a1"][s][(3, "per_channel")] for s in SEEDS) / 3
    ax = axes[1]
    ax.axhline(base_4pc, color="#888888", linewidth=0.8, linestyle=":")
    ax.annotate(
        "baseline @ 4-bit", (2.15, base_4pc), fontsize=6.5, color="#555555",
        va="bottom",
    )
    ax.annotate(
        "A1 @ 3-bit", (3, a1_3pc), textcoords="offset points", xytext=(6, -11),
        fontsize=6.5, color=COLORS["a1"],
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_quant.pdf")
    plt.close(fig)


def fig_dose() -> None:
    sweep = json.loads(
        (DATA / "phase1_exitgate" / "sweep_metrics.json").read_text()
    )["results"]
    by_label = {r["label"]: r["eval_metrics"] for r in sweep}
    base = by_label["baseline"]
    lams = [0.001, 0.01, 0.1, 1.0]
    aniso = [by_label[f"sigreg_l{lam}"]["anisotropy_h"] for lam in lams]
    dppl = [
        100.0 * (by_label[f"sigreg_l{lam}"]["perplexity"] / base["perplexity"] - 1.0)
        for lam in lams
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.2))
    ax1.plot(lams, aniso, marker="o", color=COLORS["baseline"])
    ax1.axhline(base["anisotropy_h"], color="#888888", linewidth=0.9, linestyle="--")
    ax1.annotate(
        "baseline", (lams[-1], base["anisotropy_h"]),
        textcoords="offset points", xytext=(0, 3), ha="right",
        fontsize=7, color="#555555",
    )
    ax1.set_ylabel("anisotropy($h$)")
    ax1.set_xlabel(r"$\lambda$ (log scale)")
    ax1.set_xscale("log")

    ax2.plot(lams, dppl, marker="o", color=COLORS["baseline"])
    ax2.axhline(0.0, color="#888888", linewidth=0.9, linestyle="--")
    ax2.set_ylabel(r"$\Delta$ppl vs baseline (%)")
    ax2.set_xlabel(r"$\lambda$ (log scale)")
    ax2.set_xscale("log")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dose.pdf")
    plt.close(fig)


def fig_perlayer() -> None:
    per_layer = json.loads(
        (DATA / "phase1_5" / "a1" / "probe_result.json").read_text()
    )["per_layer"]
    layers = list(range(len(per_layer["k_aniso_before"])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.4))
    spec = [
        ("k", COLORS["baseline"], "K"),
        ("v", COLORS["sigreg"], "V"),
    ]
    for ax, metric, ylabel in (
        (ax1, "aniso", "pairwise-cosine anisotropy"),
        (ax2, "top1", "spectral top-1 share"),
    ):
        for prefix, color, name in spec:
            ax.plot(
                layers, per_layer[f"{prefix}_{metric}_before"],
                marker="o", color=color, label=f"{name} before",
            )
            ax.plot(
                layers, per_layer[f"{prefix}_{metric}_after"],
                marker="o", markerfacecolor="none", color=color,
                linestyle="--", label=f"{name} after",
            )
        ax.set_xlabel("layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(layers[::2])
    ax1.legend(frameon=False, ncol=2, columnspacing=1.0, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_perlayer.pdf")
    plt.close(fig)


def main() -> None:
    global DATA, OUT
    figures = {"fig_quant": fig_quant, "fig_dose": fig_dose,
               "fig_perlayer": fig_perlayer}
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default=str(DATA),
                    help="root of the result JSONs: the output tree (default) "
                         "or an extracted artifact bundle")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--only", choices=sorted(figures), action="append",
                    help="figure(s) to produce; default: all three")
    args = ap.parse_args()
    DATA = Path(args.artifacts).resolve()
    OUT = Path(args.out_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.only or sorted(figures):
        figures[name]()
        print(f"[figures] wrote {OUT / (name + '.pdf')}")


if __name__ == "__main__":
    main()
