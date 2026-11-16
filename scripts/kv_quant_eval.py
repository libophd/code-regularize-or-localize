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

from jepa_lm.data.fineweb import build_fineweb_factories, load_fineweb_streaming
from jepa_lm.eval.evaluator import evaluate
from jepa_lm.systems.kv_quant import KVQuantize, PostRopeKQuantize
from jepa_lm.token_io.lm_heads import LinearLMHead
from jepa_lm.training.checkpoint import (
    load_checkpoint,
    load_checkpoint_blob,
)
from jepa_lm.trunks.llama import HFLlamaTrunk
from phase1_real_run import load_tokenizer_with_vocab_pin

DEFAULT_CONFIGS: list[dict[str, Any]] = [
    {"bits": None},
    *[
        {"bits": b, "granularity": g, "targets": "both"}
        for b in (8, 4, 3, 2)
        for g in ("per_token", "per_channel")
    ],
    {"bits": 3, "granularity": "per_token", "targets": "k"},
    {"bits": 3, "granularity": "per_token", "targets": "v"},
]

CONTROL_CONFIGS: list[dict[str, Any]] = [
    {"bits": None},
    *[
        {"bits": b, "granularity": g, "targets": "both", "mode": "asymmetric"}
        for b in (4, 3, 2)
        for g in ("per_token", "per_channel")
    ],
    *[
        {"bits": b, "granularity": g, "targets": "both", "subtract_mean": True}
        for b in (4, 3, 2)
        for g in ("per_token", "per_channel")
    ],
]

KIVI_CONFIGS: list[dict[str, Any]] = [
    {"bits": None},
    *[
        {"bits": b, "granularity": "mixed", "targets": "both", "mode": m}
        for b in (4, 3)
        for m in ("symmetric", "asymmetric")
    ],
    *[
        {
            "bits": b, "granularity": "mixed", "targets": "both",
            "mode": "asymmetric", "group_size": 32,
        }
        for b in (4, 3)
    ],
    *[
        {
            "bits": b, "granularity": g, "targets": "both",
            "mode": m, "group_size": 32,
        }
        for b in (4, 3)
        for g in ("per_token", "per_channel")
        for m in ("symmetric", "asymmetric")
    ],
]

RETROFIT_CONFIGS: list[dict[str, Any]] = [
    {"bits": None},
    *[
        {"bits": b, "granularity": g, "targets": "both", "rotate": True}
        for b in (4, 3)
        for g in ("per_token", "per_channel")
    ],
    *[
        {
            "bits": 3, "granularity": g, "targets": "both",
            "rotate": True, "mode": "asymmetric",
        }
        for g in ("per_token", "per_channel")
    ],
    *[
        {"bits": b, "granularity": g, "targets": "both", "standardize": True}
        for b in (4, 3)
        for g in ("per_token", "per_channel")
    ],
    *[
        {
            "bits": 3, "granularity": g, "targets": "both",
            "standardize": True, "mode": "asymmetric",
        }
        for g in ("per_token", "per_channel")
    ],
    *[
        {
            "bits": 3, "granularity": g, "targets": "both",
            "rotate": True, "standardize": True,
        }
        for g in ("per_token", "per_channel")
    ],
]

KIVI_2BIT_CONFIGS: list[dict[str, Any]] = [
    {"bits": None},
    {"bits": 2, "granularity": "mixed", "targets": "both"},
    {"bits": 2, "granularity": "mixed", "targets": "both", "mode": "asymmetric"},
    {
        "bits": 2, "granularity": "mixed", "targets": "both",
        "mode": "asymmetric", "group_size": 32,
    },
    *[
        {
            "bits": 2, "granularity": g, "targets": "both",
            "mode": m, "group_size": 32,
        }
        for g in ("per_token", "per_channel")
        for m in ("symmetric", "asymmetric")
    ],
]

GCONTROL_CONFIGS: list[dict[str, Any]] = [
    {"bits": None},
    *[
        {
            "bits": 3, "granularity": "per_channel", "targets": "both",
            "mode": m, "group_size": 2048,
        }
        for m in ("symmetric", "asymmetric")
    ],
    {
        "bits": 3, "granularity": "mixed", "targets": "both",
        "mode": "symmetric", "group_size": 32,
    },
]

POSTROPE_CONFIGS: list[dict[str, Any]] = [
    {"bits": None},
    *[
        {"bits": b, "granularity": g, "targets": "both", "locus": "post_rope_k"}
        for b in (4, 3)
        for g in ("per_token", "per_channel")
    ],
    *[
        {
            "bits": 3, "granularity": g, "targets": "both",
            "mode": "asymmetric", "locus": "post_rope_k",
        }
        for g in ("per_token", "per_channel")
    ],
]

POSTROPE_KIVI_CONFIGS: list[dict[str, Any]] = [
    {"bits": None},
    *[
        {
            "bits": b, "granularity": "mixed", "targets": "both",
            "mode": "asymmetric", "group_size": 32, "locus": "post_rope_k",
        }
        for b in (4, 3)
    ],
    {
        "bits": 3, "granularity": "per_token", "targets": "both",
        "mode": "asymmetric", "group_size": 32, "locus": "post_rope_k",
    },
    {
        "bits": 3, "granularity": "per_channel", "targets": "both",
        "group_size": 32, "locus": "post_rope_k",
    },
]

PERHEAD_CONFIGS: list[dict[str, Any]] = [
    {"bits": None},
    *[
        {
            "bits": b, "granularity": "per_token", "targets": "both",
            "mode": m, "group_size": 64,
        }
        for b in (4, 3)
        for m in ("symmetric", "asymmetric")
    ],
]

CONFIG_SETS = {
    "default": DEFAULT_CONFIGS,
    "controls": CONTROL_CONFIGS,
    "kivi": KIVI_CONFIGS,
    "kivi_2bit": KIVI_2BIT_CONFIGS,
    "retrofit": RETROFIT_CONFIGS,
    "gcontrol": GCONTROL_CONFIGS,
    "postrope": POSTROPE_CONFIGS,
    "postrope_kivi": POSTROPE_KIVI_CONFIGS,
    "perhead": PERHEAD_CONFIGS,
}


def quantizer_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    common: dict[str, Any] = {
        "bits": cfg["bits"],
        "mode": cfg.get("mode", "symmetric"),
        "subtract_mean": bool(cfg.get("subtract_mean", False)),
        "group_size": cfg.get("group_size"),
        "standardize": bool(cfg.get("standardize", False)),
        "rotate": bool(cfg.get("rotate", False)),
    }
    if cfg["granularity"] == "mixed":
        if cfg["targets"] != "both":
            raise ValueError("mixed granularity implies targets='both'")
        return [
            {"granularity": "per_channel", "targets": "k", **common},
            {"granularity": "per_token", "targets": "v", **common},
        ]
    return [{"granularity": cfg["granularity"], "targets": cfg["targets"], **common}]


def quantizer_contexts(trunk: Any, cfg: dict[str, Any]) -> list[Any]:
    if cfg.get("locus") == "post_rope_k":
        common = {
            "bits": cfg["bits"],
            "mode": cfg.get("mode", "symmetric"),
            "group_size": cfg.get("group_size"),
        }
        if cfg["granularity"] == "mixed":
            k_gran, v_gran = "per_channel", "per_token"
        else:
            k_gran = v_gran = cfg["granularity"]
        ctxs: list[Any] = [PostRopeKQuantize(granularity=k_gran, **common)]
        if cfg.get("targets", "both") in ("v", "both"):
            ctxs.append(
                KVQuantize(trunk, granularity=v_gran, targets="v", **common)
            )
        return ctxs
    return [KVQuantize(trunk, **spec) for spec in quantizer_specs(cfg)]


def load_model(path: Path) -> tuple[torch.nn.Module, torch.nn.Module, str]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return load_model_blob(blob)


def load_model_blob(blob) -> tuple[torch.nn.Module, torch.nn.Module, str]:
    if isinstance(blob, dict) and blob.get("format") == "a1_adapted_trunk_v1":
        trunk = HFLlamaTrunk(**blob["trunk_kwargs"])
        trunk.load_state_dict(blob["trunk"])
        head = LinearLMHead(
            blob["trunk_kwargs"]["hidden_size"], blob["trunk_kwargs"]["vocab_size"]
        )
        head.load_state_dict(blob["head"])
        if blob.get("tie_lm_head"):
            head.tie_to(trunk.get_input_embeddings())
        return trunk, head, "a1_adapted"
    trainer = load_checkpoint_blob(blob)
    kind = "+".join(t.name for t in trainer.config.loss_terms)
    return trainer.trunk, trainer.head, kind


def config_label(cfg: dict[str, Any]) -> str:
    if cfg["bits"] is None:
        return "fp32-ref"
    label = f"{cfg['bits']}bit-{cfg['granularity']}-{cfg['targets']}"
    if cfg.get("mode", "symmetric") != "symmetric":
        label += "-asym"
    if cfg.get("subtract_mean"):
        label += "-meansub"
    if cfg.get("group_size"):
        label += f"-g{cfg['group_size']}"
    if cfg.get("standardize"):
        label += "-std"
    if cfg.get("rotate"):
        label += "-rot"
    if cfg.get("locus") == "post_rope_k":
        label += "-postrope"
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description="KV-quantization NLL sweep.")
    parser.add_argument("models", nargs="+", help="checkpoint or adapted_trunk .pt paths")
    parser.add_argument("--tokenizer-name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--eval-tokens", type=int, default=2_000_000)
    parser.add_argument("--eval-docs", type=int, default=5000)
    parser.add_argument("--eval-offset-docs", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--eval-seed", type=int, default=99)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", default=None)
    parser.add_argument(
        "--config-set", default="default", choices=sorted(CONFIG_SETS),
        help="'default' = headline symmetric grid + K/V diagnostics; "
        "'controls' = mean-absorbing asymmetric + mean-subtraction arms.",
    )
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
            raise SystemExit(f"[kv-quant] model not found: {path}")
        trunk, head, kind = load_model(path)
        trunk.to(device).eval()
        head.to(device).eval()
        print(f"[kv-quant] === {path} (kind: {kind}) ===")

        reference_nll: float | None = None
        for cfg in CONFIG_SETS[args.config_set]:
            label = config_label(cfg)
            torch.manual_seed(args.eval_seed)
            t0 = time.time()
            if cfg["bits"] is None:
                metrics = evaluate(trunk, head, eval_factory(), device=device)
            else:
                with ExitStack() as stack:
                    for ctx in quantizer_contexts(trunk, cfg):
                        stack.enter_context(ctx)
                    metrics = evaluate(trunk, head, eval_factory(), device=device)
            nll = metrics["mean_nll"]
            if cfg["bits"] is None:
                reference_nll = nll
            delta = nll - reference_nll if reference_nll is not None else float("nan")
            row = {
                "model": str(path),
                "kind": kind,
                "config": label,
                **{k: v for k, v in cfg.items()},
                "mean_nll": nll,
                "perplexity": metrics["perplexity"],
                "nll_delta_vs_ref": delta,
                "anisotropy_h": metrics["anisotropy_h"],
                "elapsed_s": round(time.time() - t0, 1),
            }
            all_results.append(row)
            print(
                f"[kv-quant] {label:>24}: nll {nll:.4f} (Δ {delta:+.4f}) "
                f"ppl {metrics['perplexity']:.3f}  [{row['elapsed_s']}s]"
            )
        trunk.to("cpu")
        head.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, indent=2))
        print(f"[kv-quant] JSON -> {out}")


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
