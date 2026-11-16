from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def derive_export_name(ckpt_path: Path) -> str:
    label = ckpt_path.parent.name
    parts = ckpt_path.parts
    if "checkpoints" in parts:
        i = parts.index("checkpoints")
        run_tag = parts[i - 1] if i > 0 and parts[i - 1] != "outputs" else parts[i + 1]
    else:
        run_tag = ckpt_path.parent.parent.name
    return f"{run_tag}__{label}"


def export_checkpoint(
    ckpt_path: Path,
    out_dir: Path,
    tokenizer_name: str,
    check_tokens: int = 64,
    check_batch: int = 2,
) -> dict:
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    from jepa_lm.token_io.lm_heads import LinearLMHead
    from jepa_lm.trunks.llama import HFLlamaTrunk

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    if cfg.trunk_name != "hf_llama" or cfg.lm_head_name != "linear":
        raise SystemExit(
            f"[export] {ckpt_path}: only hf_llama + linear head checkpoints are "
            f"supported; got trunk={cfg.trunk_name!r} head={cfg.lm_head_name!r}."
        )

    trunk_kwargs = dict(cfg.trunk_kwargs or {})
    hf_config = LlamaConfig(
        **trunk_kwargs,
        use_cache=True,
        tie_word_embeddings=bool(cfg.tie_lm_head),
    )
    lm = LlamaForCausalLM(hf_config)

    trunk_sd = blob["trunk"]
    stripped = {}
    for key, value in trunk_sd.items():
        if not key.startswith("model."):
            raise SystemExit(
                f"[export] {ckpt_path}: unexpected trunk key {key!r} (no 'model.' "
                f"prefix) — trunk layout drifted from HFLlamaTrunk."
            )
        stripped[key[len("model."):]] = value
    lm.model.load_state_dict(stripped, strict=True)

    head_weight = blob["head"]["proj.weight"]
    if cfg.tie_lm_head:
        if not torch.equal(head_weight, stripped["embed_tokens.weight"]):
            raise SystemExit(
                f"[export] {ckpt_path}: tie_lm_head=True but head weight != "
                f"embedding weight — checkpoint is not what it claims."
            )
        lm.tie_weights()
    else:
        lm.lm_head.load_state_dict({"weight": head_weight})

    trunk = HFLlamaTrunk(**trunk_kwargs)
    trunk.load_state_dict(trunk_sd)
    head = LinearLMHead(cfg.trunk_kwargs["hidden_size"], cfg.trunk_kwargs["vocab_size"])
    head.load_state_dict(blob["head"])
    trunk.eval(), head.eval(), lm.eval()
    torch.manual_seed(0)
    input_ids = torch.randint(
        0, cfg.trunk_kwargs["vocab_size"], (check_batch, check_tokens)
    )
    with torch.no_grad():
        reference = head(trunk(input_ids))
        exported = lm(input_ids=input_ids).logits
    if not torch.allclose(reference, exported, rtol=1e-4, atol=1e-5):
        max_err = (reference - exported).abs().max().item()
        raise SystemExit(
            f"[export] {ckpt_path}: exported logits diverge from the original "
            f"trunk+head (max abs err {max_err:.3e}) — refusing to save."
        )
    max_err = (reference - exported).abs().max().item()

    out_dir.mkdir(parents=True, exist_ok=True)
    lm.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(out_dir)

    manifest = {
        "source_checkpoint": str(ckpt_path),
        "global_step": int(blob["global_step"]),
        "trainer_seed": int(cfg.seed),
        "loss_terms": [t.name for t in cfg.loss_terms],
        "tokenizer": tokenizer_name,
        "tie_word_embeddings": bool(cfg.tie_lm_head),
        "logits_max_abs_err": max_err,
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Phase 1 checkpoints to HF LlamaForCausalLM dirs."
    )
    parser.add_argument("checkpoints", nargs="+", help="checkpoint .pt paths")
    parser.add_argument("--out-root", default="outputs/hf_export")
    parser.add_argument("--tokenizer", default="mistralai/Mistral-7B-v0.1")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    for raw in args.checkpoints:
        ckpt_path = Path(raw)
        if not ckpt_path.is_file():
            raise SystemExit(f"[export] checkpoint not found: {ckpt_path}")
        name = derive_export_name(ckpt_path)
        manifest = export_checkpoint(ckpt_path, out_root / name, args.tokenizer)
        print(
            f"[export] {name}: step {manifest['global_step']} "
            f"seed {manifest['trainer_seed']} terms {manifest['loss_terms']} "
            f"logits_err {manifest['logits_max_abs_err']:.2e} -> {out_root / name}"
        )


if __name__ == "__main__":
    main()
