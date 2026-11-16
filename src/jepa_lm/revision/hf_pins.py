from __future__ import annotations

import os
from pathlib import Path

PINNED_HF_REVISIONS = {
    "mistralai/Mistral-7B-v0.1": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
    "HuggingFaceFW/fineweb": "9bb295ddab0e05d785b879661af7260fed5140fc",
}


def hf_cache_dir() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:
        if os.environ.get("HF_HUB_CACHE"):
            return Path(os.environ["HF_HUB_CACHE"])
        if os.environ.get("HF_HOME"):
            return Path(os.environ["HF_HOME"]) / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_cached_revision(
    kind: str, repo_id: str, *, expected: str | None = None,
) -> str:
    ref = hf_cache_dir() / f"{kind}--{repo_id.replace('/', '--')}" / "refs" / "main"
    try:
        got = ref.read_text().strip()
    except OSError:
        raise SystemExit(
            f"[hf-pins] {repo_id} is not in the HF cache at {hf_cache_dir()} — "
            "prefetch it (warm cache is part of the pinned protocol), then relaunch"
        ) from None
    want = PINNED_HF_REVISIONS.get(repo_id) if expected is None else expected
    if want is not None and got != want:
        raise SystemExit(
            f"[hf-pins] {repo_id} cache resolves to {got}, but the registered "
            f"protocol pins {want} — refusing to run against a drifted revision"
        )
    return got


def load_pinned_tokenizer(name: str, revision: str, expected_vocab: int):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    if tok.vocab_size != expected_vocab:
        raise SystemExit(f"[hf-pins] tokenizer vocab_size {tok.vocab_size} != {expected_vocab}")
    if len(tok) != expected_vocab:
        raise SystemExit(f"[hf-pins] len(tokenizer) {len(tok)} != {expected_vocab}")
    if tok.eos_token_id is None or tok.eos_token_id >= expected_vocab:
        raise SystemExit(f"[hf-pins] bad eos_token_id {tok.eos_token_id}")
    return tok


def load_pinned_fineweb(name: str, config: str, split: str, revision: str):
    from datasets import load_dataset

    return load_dataset(name, config, split=split, streaming=True, revision=revision)
