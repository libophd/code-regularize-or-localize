from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import Tensor

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def average_ranks(x: Tensor) -> Tensor:
    order = torch.argsort(x)
    sorted_x = x[order]
    n = x.numel()
    ranks_sorted = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks_sorted[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = ranks_sorted
    return ranks


def spearman(a: Tensor, b: Tensor) -> float:
    if a.numel() != b.numel() or a.numel() < 2:
        raise ValueError(f"spearman needs two equal-length vectors, n>=2; got {a.numel()}, {b.numel()}")
    ra, rb = average_ranks(a.flatten()), average_ranks(b.flatten())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if denom == 0:
        raise ValueError("spearman undefined: a rank vector is constant")
    return float((ra @ rb) / denom)


def all_but_the_top(embeddings: Tensor, top_d: int) -> Tensor:
    if top_d < 0:
        raise ValueError(f"top_d must be >= 0, got {top_d}")
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    if top_d == 0:
        return centered
    _u, _s, vh = torch.linalg.svd(centered.double(), full_matrices=False)
    top = vh[:top_d]
    return (centered.double() - (centered.double() @ top.T) @ top).to(embeddings.dtype)


def embed_sentences(
    model, tokenizer, sentences: list[str], device: torch.device, batch_size: int = 64
) -> Tensor:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start : start + batch_size]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            rows.append(pooled.float().cpu())
    return torch.cat(rows, dim=0)


def evaluate_model(model_dir: Path, device: torch.device, top_d: int) -> dict:
    from transformers import AutoModel, AutoTokenizer

    from jepa_lm.eval.metrics import anisotropy

    import datasets as hf_datasets

    stsb = hf_datasets.load_dataset("glue", "stsb", split="validation")
    gold = torch.tensor(stsb["label"], dtype=torch.float64)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(model_dir).to(device)

    emb1 = embed_sentences(model, tokenizer, list(stsb["sentence1"]), device)
    emb2 = embed_sentences(model, tokenizer, list(stsb["sentence2"]), device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def sts_and_aniso(e1: Tensor, e2: Tensor) -> tuple[float, float]:
        cos = torch.nn.functional.cosine_similarity(e1, e2, dim=-1).double()
        rho = spearman(cos, gold)
        aniso = float(anisotropy(torch.cat([e1, e2], dim=0)))
        return rho, aniso

    rho_raw, aniso_raw = sts_and_aniso(emb1, emb2)
    corpus = torch.cat([emb1, emb2], dim=0)
    whitened = all_but_the_top(corpus, top_d)
    w1, w2 = whitened[: emb1.shape[0]], whitened[emb1.shape[0] :]
    rho_white, aniso_white = sts_and_aniso(w1, w2)

    return {
        "model": model_dir.name,
        "n_pairs": int(gold.numel()),
        "top_d": top_d,
        "sts_spearman": rho_raw,
        "sts_spearman_whitened": rho_white,
        "anisotropy_pooled": aniso_raw,
        "anisotropy_pooled_whitened": aniso_white,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STS-B + All-but-the-Top comparator over exported HF models."
    )
    parser.add_argument("models", nargs="+", help="HF export dirs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--top-d", type=int, default=None,
        help="principal components to remove (default: max(1, d_model//100), Mu et al.)",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    results = []
    for raw in args.models:
        model_dir = Path(raw)
        if not (model_dir / "config.json").is_file():
            raise SystemExit(f"[sts] not an HF export dir: {model_dir}")
        d_model = json.loads((model_dir / "config.json").read_text())["hidden_size"]
        top_d = args.top_d if args.top_d is not None else max(1, d_model // 100)
        r = evaluate_model(model_dir, device, top_d)
        results.append(r)
        print(
            f"[sts] {r['model']}: spearman {r['sts_spearman']:.4f} "
            f"(whitened {r['sts_spearman_whitened']:.4f}, top_d={r['top_d']}) | "
            f"pooled aniso {r['anisotropy_pooled']:.4f} "
            f"(whitened {r['anisotropy_pooled_whitened']:.4f})"
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"[sts] JSON written to {out}")


if __name__ == "__main__":
    main()
