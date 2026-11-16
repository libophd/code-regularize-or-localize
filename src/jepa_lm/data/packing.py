from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence


def pack_token_stream(
    token_stream: Iterable[Sequence[int]],
    seq_len: int,
    *,
    eos_token_id: int | None = None,
    drop_last: bool = True,
) -> Iterator[dict[str, list[int]]]:
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2; got {seq_len}")

    buf: list[int] = []
    for doc_ids in token_stream:
        buf.extend(doc_ids)
        if eos_token_id is not None:
            buf.append(eos_token_id)
        while len(buf) >= seq_len:
            yield {"input_ids": buf[:seq_len]}
            buf = buf[seq_len:]
    if not drop_last and buf:
        yield {"input_ids": buf}
