from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import torch

from jepa_lm.data.fineweb import build_fineweb_factories, load_fineweb_streaming
from jepa_lm.eval.channel_stats import (
    RunningChannelMoments,
    RunningCovariance,
    axis_alignment,
    scale_dispersion,
    token_norm_concentration,
)
from jepa_lm.systems.kv_capture import KVCapture
from kv_quant_eval import load_model
from phase1_real_run import load_tokenizer_with_vocab_pin


def collect_stats(
    trunk: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    max_tokens: int,
) -> list[dict[str, Any]]:
    moments: dict[tuple[int, str], RunningChannelMoments] = {}
    covs: dict[tuple[int, str], RunningCovariance] = {}
    sink_ratios: dict[tuple[int, str], list[float]] = {}
    seen = 0
    with KVCapture(trunk) as cap, torch.no_grad():
        for raw_batch in dataloader:
            input_ids = raw_batch["input_ids"].to(device)
            trunk(input_ids)
            k_layers, v_layers = cap.captured()
            for layer, (k, v) in enumerate(zip(k_layers, v_layers)):
                for name, tensor in (("k", k), ("v", v)):
                    flat = tensor.reshape(-1, tensor.shape[-1]).float()
                    key = (layer, name)
                    if key not in moments:
                        moments[key] = RunningChannelMoments(flat.shape[-1])
                        covs[key] = RunningCovariance(flat.shape[-1])
                        sink_ratios[key] = []
                    moments[key].update(flat)
                    covs[key].update(flat)
                    sink_ratios[key].append(token_norm_concentration(flat))
            cap.reset()
            seen += input_ids.numel()
            if seen >= max_tokens:
                break
    rows = []
    for (layer, name), rm in sorted(moments.items()):
        kurt = rm.excess_kurtosis()
        ratios = torch.tensor(sink_ratios[(layer, name)])
        rows.append({
            "layer": layer,
            "tensor": name,
            "scale_dispersion": scale_dispersion(rm.std()),
            "kurtosis_med": kurt.median().item(),
            "kurtosis_max": kurt.max().item(),
            "sink_ratio": ratios.median().item(),
            "axis_alignment": axis_alignment(covs[(layer, name)].covariance()),
            "mean_abs_mean": rm.mean().abs().mean().item(),
            "n_tokens": rm.n,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Channel-level K/V outlier statistics.")
    parser.add_argument("models", nargs="+", help="checkpoint or adapted_trunk .pt paths")
    parser.add_argument("--tokenizer-name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--eval-tokens", type=int, default=500_000)
    parser.add_argument("--eval-docs", type=int, default=5000)
    parser.add_argument("--eval-offset-docs", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--eval-seed", type=int, default=99)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = load_tokenizer_with_vocab_pin(args.tokenizer_name, expected_vocab=32000)

    def raw_stream_factory():
        return load_fineweb_streaming(args.dataset_name, args.dataset_config, split="train")

    def tokenize_fn(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    n_eval_batches = math.ceil(args.eval_tokens / (args.batch_size * args.seq_len))
    _train_factory, eval_factory = build_fineweb_factories(
        raw_stream_factory,
        tokenize_fn,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        eos_token_id=tokenizer.eos_token_id,
        eval_docs=args.eval_docs,
        eval_offset_docs=args.eval_offset_docs,
        eval_n_batches=n_eval_batches,
    )

    all_results: list[dict[str, Any]] = []
    for raw in args.models:
        path = Path(raw)
        if not path.is_file():
            raise SystemExit(f"[channel-stats] model not found: {path}")
        trunk, _head, kind = load_model(path)
        trunk.to(device).eval()
        print(f"[channel-stats] === {path} (kind: {kind}) ===")
        torch.manual_seed(args.eval_seed)
        t0 = time.time()
        rows = collect_stats(
            trunk, eval_factory(), device=device, max_tokens=args.eval_tokens
        )
        for row in rows:
            row.update({"model": str(path), "kind": kind})
            all_results.append(row)
        for name in ("k", "v"):
            sub = [r for r in rows if r["tensor"] == name]
            print(
                f"[channel-stats] {name.upper()}: disp med "
                f"{sorted(r['scale_dispersion'] for r in sub)[len(sub)//2]:.2f}, "
                f"kurt med {sorted(r['kurtosis_med'] for r in sub)[len(sub)//2]:.2f}, "
                f"sink med {sorted(r['sink_ratio'] for r in sub)[len(sub)//2]:.2f}, "
                f"align med {sorted(r['axis_alignment'] for r in sub)[len(sub)//2]:.2f} "
                f"[{time.time() - t0:.0f}s]"
            )
        trunk.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, indent=2))
        print(f"[channel-stats] JSON -> {out}")


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
