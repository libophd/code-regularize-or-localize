from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch.utils.data

from jepa_lm.data.collator import PackedLMCollator
from jepa_lm.data.packing import pack_token_stream


def tokenize_stream(
    examples: Iterable[Mapping[str, Any]],
    tokenize_fn: Callable[[str], Sequence[int]],
    *,
    text_key: str = "text",
) -> Iterator[Sequence[int]]:
    for example in examples:
        text = example[text_key]
        yield tokenize_fn(text)


class IterableTokenStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self, stream_factory: Callable[[], Iterable[Mapping[str, Any]]]) -> None:
        super().__init__()
        self.stream_factory = stream_factory

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self.stream_factory())


def build_packed_lm_dataloader(
    raw_stream_factory: Callable[[], Iterable[Mapping[str, Any]]],
    tokenize_fn: Callable[[str], Sequence[int]],
    *,
    seq_len: int,
    batch_size: int,
    text_key: str = "text",
    eos_token_id: int | None = None,
    drop_last: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    def packed_stream_factory() -> Iterator[Mapping[str, Any]]:
        raw = raw_stream_factory()
        tokens = tokenize_stream(raw, tokenize_fn, text_key=text_key)
        return pack_token_stream(
            tokens, seq_len=seq_len, eos_token_id=eos_token_id, drop_last=drop_last
        )

    dataset = IterableTokenStreamDataset(packed_stream_factory)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(0)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=PackedLMCollator(seq_len=seq_len),
        num_workers=num_workers,
        generator=loader_generator,
    )
