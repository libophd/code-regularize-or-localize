from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from jepa_lm.data.streams import build_packed_lm_dataloader


def split_streaming_dataset(
    raw_stream: Any,
    *,
    eval_docs: int,
    eval_offset_docs: int = 0,
) -> tuple[Any, Any]:
    if eval_docs < 1:
        raise ValueError(f"eval_docs must be >= 1; got {eval_docs}")
    if eval_offset_docs < 0:
        raise ValueError(f"eval_offset_docs must be >= 0; got {eval_offset_docs}")
    eval_stream = raw_stream.skip(eval_offset_docs).take(eval_docs)
    train_stream = raw_stream.skip(eval_offset_docs + eval_docs)
    return train_stream, eval_stream


def load_fineweb_streaming(
    dataset_name: str = "HuggingFaceFW/fineweb",
    dataset_config: str | None = "sample-10BT",
    *,
    split: str = "train",
) -> Any:
    from datasets import load_dataset

    return load_dataset(
        dataset_name, name=dataset_config, split=split, streaming=True
    )


def build_fineweb_factories(
    raw_stream_factory: Callable[[], Any],
    tokenize_fn: Callable[[str], Sequence[int]],
    *,
    seq_len: int,
    batch_size: int,
    eos_token_id: int,
    eval_docs: int,
    eval_n_batches: int,
    eval_offset_docs: int = 0,
    text_key: str = "text",
    num_workers: int = 0,
) -> tuple[Callable[[], Iterable], Callable[[], Iterator]]:
    if eval_docs < 1 or eval_n_batches < 1:
        raise ValueError(
            f"eval_docs and eval_n_batches must be >= 1; got "
            f"eval_docs={eval_docs}, eval_n_batches={eval_n_batches}"
        )

    def _train_stream_factory() -> Iterable[Mapping[str, Any]]:
        full = raw_stream_factory()
        train_sub, _ = split_streaming_dataset(
            full, eval_docs=eval_docs, eval_offset_docs=eval_offset_docs
        )
        return train_sub

    def _eval_stream_factory() -> Iterable[Mapping[str, Any]]:
        full = raw_stream_factory()
        _, eval_sub = split_streaming_dataset(
            full, eval_docs=eval_docs, eval_offset_docs=eval_offset_docs
        )
        return eval_sub

    train_loader = build_packed_lm_dataloader(
        _train_stream_factory,
        tokenize_fn,
        seq_len=seq_len,
        batch_size=batch_size,
        text_key=text_key,
        eos_token_id=eos_token_id,
        drop_last=True,
        num_workers=num_workers,
    )
    eval_loader = build_packed_lm_dataloader(
        _eval_stream_factory,
        tokenize_fn,
        seq_len=seq_len,
        batch_size=batch_size,
        text_key=text_key,
        eos_token_id=eos_token_id,
        drop_last=True,
        num_workers=num_workers,
    )

    def train_factory() -> Iterable:
        return train_loader

    def eval_factory() -> Iterator:
        return itertools.islice(eval_loader, eval_n_batches)

    return train_factory, eval_factory
