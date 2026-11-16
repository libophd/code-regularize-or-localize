from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import torch
import torch.nn.functional as F

BOOT_CONFIGS: list[dict[str, Any]] = [
    {"bits": None},
    {"bits": 3, "granularity": "per_token", "targets": "both"},
    {"bits": 3, "granularity": "per_channel", "targets": "both"},
    {"bits": 3, "granularity": "per_token", "targets": "both", "mode": "asymmetric"},
    {"bits": 3, "granularity": "per_channel", "targets": "both", "mode": "asymmetric"},
    {"bits": 3, "granularity": "per_token", "targets": "both", "subtract_mean": True},
    {"bits": 3, "granularity": "per_channel", "targets": "both", "subtract_mean": True},
    {"bits": 3, "granularity": "mixed", "targets": "both"},
    {"bits": 3, "granularity": "mixed", "targets": "both", "mode": "asymmetric"},
    {"bits": 3, "granularity": "per_token", "targets": "both", "rotate": True},
    {"bits": 3, "granularity": "per_token", "targets": "both", "mode": "asymmetric", "rotate": True},
    {"bits": 3, "granularity": "per_token", "targets": "both", "standardize": True, "rotate": True},
    {"bits": 4, "granularity": "per_token", "targets": "both"},
    {"bits": 4, "granularity": "per_channel", "targets": "both"},
    {"bits": 4, "granularity": "per_token", "targets": "both", "mode": "asymmetric"},
    {"bits": 4, "granularity": "per_channel", "targets": "both", "mode": "asymmetric"},
    {"bits": 4, "granularity": "per_token", "targets": "both", "subtract_mean": True},
    {"bits": 4, "granularity": "per_channel", "targets": "both", "subtract_mean": True},
    {"bits": 4, "granularity": "mixed", "targets": "both"},
    {"bits": 4, "granularity": "mixed", "targets": "both", "mode": "asymmetric"},
]

TIERS = {
    "orig3": lambda label: label.startswith("3bit") and "mixed" not in label
    and "rot" not in label and "std" not in label,
    "ext3": lambda label: label.startswith("3bit"),
    "orig4": lambda label: label.startswith("4bit") and "mixed" not in label,
    "m4": lambda label: label.startswith("4bit"),
}


def per_batch_nll(trunk, head, dataloader, device) -> tuple[list[float], list[int]]:
    nll_sums: list[float] = []
    token_counts: list[int] = []
    with torch.no_grad():
        for raw_batch in dataloader:
            input_ids = raw_batch["input_ids"].to(device)
            labels = raw_batch.get("labels", input_ids).to(device)
            logits = head(trunk(input_ids))
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid = shift_labels != -100
            per_token = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                torch.where(valid, shift_labels, torch.full_like(shift_labels, -100)).reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            valid_f = valid.reshape(-1).to(per_token.dtype)
            nll_sums.append(float((per_token * valid_f).sum().item()))
            token_counts.append(int(valid_f.sum().item()))
    return nll_sums, token_counts


def cmd_run(args: argparse.Namespace) -> None:
    from jepa_lm.data.fineweb import build_fineweb_factories, load_fineweb_streaming
    from kv_quant_eval import config_label, load_model, quantizer_contexts
    from phase1_real_run import load_tokenizer_with_vocab_pin

    device = torch.device(args.device)
    tokenizer = load_tokenizer_with_vocab_pin(args.tokenizer_name, expected_vocab=32000)

    def raw_stream_factory():
        return load_fineweb_streaming(args.dataset_name, args.dataset_config, split="train")

    n_eval_batches = math.ceil(args.eval_tokens / (args.batch_size * args.seq_len))
    _train, eval_factory = build_fineweb_factories(
        raw_stream_factory,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        eos_token_id=tokenizer.eos_token_id,
        eval_docs=args.eval_docs,
        eval_offset_docs=args.eval_offset_docs,
        eval_n_batches=n_eval_batches,
    )

    records: list[dict[str, Any]] = []
    for raw in args.models:
        path = Path(raw)
        if not path.is_file():
            raise SystemExit(f"[bootstrap] model not found: {path}")
        trunk, head, kind = load_model(path)
        trunk.to(device).eval()
        head.to(device).eval()
        print(f"[bootstrap] === {path} (kind: {kind}) ===")
        for cfg in BOOT_CONFIGS:
            label = config_label(cfg)
            torch.manual_seed(args.eval_seed)
            t0 = time.time()
            with ExitStack() as stack:
                if cfg["bits"] is not None:
                    for ctx in quantizer_contexts(trunk, cfg):
                        stack.enter_context(ctx)
                nll_sums, token_counts = per_batch_nll(trunk, head, eval_factory(), device)
            mean = sum(nll_sums) / sum(token_counts)
            records.append({
                "model": str(path), "kind": kind, "config": label,
                "nll_sums": nll_sums, "token_counts": token_counts,
                "mean_nll": mean,
            })
            print(f"[bootstrap] {label:>28}: nll {mean:.4f} over "
                  f"{len(nll_sums)} batches  [{time.time() - t0:.1f}s]")
        trunk.to("cpu")
        head.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1))
    print(f"[bootstrap] records -> {out}")


def paired_bootstrap(
    base_rows: dict[str, tuple[list[float], list[int]]],
    a1_rows: dict[str, tuple[list[float], list[int]]],
    tier_labels: list[str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    n_batches = len(next(iter(base_rows.values()))[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_batches, size=(n_boot, n_batches))

    def matrices(rows, labels):
        sums = np.array([rows[c][0] for c in labels])
        toks = np.array([rows[c][1] for c in labels], dtype=float)
        return sums, toks

    def replicate_nll(sums, toks):
        s = sums[:, idx].sum(axis=2)
        t = toks[:, idx].sum(axis=2)
        return (s / t).T

    out: dict[str, Any] = {}
    for accounting in ("relative", "absolute"):
        b_nll = replicate_nll(*matrices(base_rows, tier_labels))
        a_nll = replicate_nll(*matrices(a1_rows, tier_labels))
        if accounting == "relative":
            ref_b = replicate_nll(*matrices(base_rows, ["fp32-ref"]))
            ref_a = replicate_nll(*matrices(a1_rows, ["fp32-ref"]))
            b_best = (b_nll - ref_b).min(axis=1)
            a_best = (a_nll - ref_a).min(axis=1)
        else:
            b_best = b_nll.min(axis=1)
            a_best = a_nll.min(axis=1)
        margins = b_best - a_best
        def full(rows):
            sums, toks = matrices(rows, tier_labels)
            return sums.sum(axis=1) / toks.sum(axis=1)
        fb, fa = full(base_rows), full(a1_rows)
        if accounting == "relative":
            ref_fb = sum(base_rows["fp32-ref"][0]) / sum(base_rows["fp32-ref"][1])
            ref_fa = sum(a1_rows["fp32-ref"][0]) / sum(a1_rows["fp32-ref"][1])
            point = (fb - ref_fb).min() - (fa - ref_fa).min()
        else:
            point = fb.min() - fa.min()
        out[accounting] = {
            "point_margin": float(point),
            "boot_se": float(margins.std(ddof=1)),
            "ci95": [float(np.percentile(margins, 2.5)), float(np.percentile(margins, 97.5))],
            "p_a1_wins": float((margins > 0).mean()),
        }
    return out


def cmd_analyze(args: argparse.Namespace) -> None:
    records = json.loads(Path(args.records).read_text())
    by_model: dict[str, dict[str, tuple[list[float], list[int]]]] = {}
    kinds: dict[str, str] = {}
    for r in records:
        by_model.setdefault(r["model"], {})[r["config"]] = (r["nll_sums"], r["token_counts"])
        kinds[r["model"]] = r["kind"]
    seeds = sorted({s for m in by_model for s in ("s42", "s43", "s44") if s in m})
    summary: dict[str, Any] = {"n_boot": args.n_boot, "seed": args.boot_seed, "tiers": {}}
    for tier, keep in TIERS.items():
        all_labels = sorted({c for m in by_model.values() for c in m if c != "fp32-ref"})
        tier_labels = [c for c in all_labels if keep(c)]
        summary["tiers"][tier] = {"configs": tier_labels, "per_seed": {}}
        for s in seeds:
            base = next(m for m in by_model if s in m and kinds[m] == "l_aux")
            a1 = next(m for m in by_model if s in m and kinds[m] == "a1_adapted")
            res = paired_bootstrap(
                by_model[base], by_model[a1], tier_labels,
                n_boot=args.n_boot, seed=args.boot_seed,
            )
            summary["tiers"][tier]["per_seed"][s] = res
            rel, ab = res["relative"], res["absolute"]
            print(
                f"[{tier} {s}] rel margin {rel['point_margin']:+.4f} "
                f"(SE {rel['boot_se']:.4f}, CI [{rel['ci95'][0]:+.4f},{rel['ci95'][1]:+.4f}], "
                f"P(A1) {rel['p_a1_wins']:.3f}) | abs margin {ab['point_margin']:+.4f} "
                f"(SE {ab['boot_se']:.4f}, CI [{ab['ci95'][0]:+.4f},{ab['ci95'][1]:+.4f}], "
                f"P(A1) {ab['p_a1_wins']:.3f})"
            )
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1))
        print(f"[bootstrap] summary -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="GPU: record per-batch NLL per model x config")
    run.add_argument("models", nargs="+")
    run.add_argument("--tokenizer-name", default="mistralai/Mistral-7B-v0.1")
    run.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    run.add_argument("--dataset-config", default="sample-10BT")
    run.add_argument("--eval-tokens", type=int, default=2_000_000)
    run.add_argument("--eval-docs", type=int, default=5000)
    run.add_argument("--eval-offset-docs", type=int, default=100_000)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--seq-len", type=int, default=2048)
    run.add_argument("--eval-seed", type=int, default=99)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--out-json", required=True)
    ana = sub.add_parser("analyze", help="CPU: paired bootstrap over the records")
    ana.add_argument("--records", required=True)
    ana.add_argument("--n-boot", type=int, default=10_000)
    ana.add_argument("--boot-seed", type=int, default=0)
    ana.add_argument("--out-json", default=None)
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
