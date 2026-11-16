from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from jepa_lm.registry import Registry

COLLATORS: Registry[Collator] = Registry("collator")


class Collator(ABC):
    @abstractmethod
    def __call__(self, examples: list[Mapping[str, Sequence[int]]]) -> dict[str, Tensor]: ...


@COLLATORS.register("packed_lm")
class PackedLMCollator(Collator):
    def __init__(self, seq_len: int) -> None:
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2; got {seq_len}")
        self.seq_len = seq_len

    def __call__(self, examples: list[Mapping[str, Sequence[int]]]) -> dict[str, Tensor]:
        if not examples:
            raise ValueError("PackedLMCollator: empty batch")
        ids_list: list[Sequence[int]] = []
        for i, ex in enumerate(examples):
            ids = ex.get("input_ids")
            if ids is None:
                raise KeyError(f"PackedLMCollator: example {i} missing 'input_ids'")
            if len(ids) != self.seq_len:
                raise ValueError(
                    f"PackedLMCollator: example {i} has length {len(ids)} != seq_len={self.seq_len}; "
                    f"caller must pre-pack via pack_token_stream (dense no-mask batches are "
                    f"load-bearing for l_sigreg)"
                )
            ids_list.append(ids)
        input_ids = torch.tensor(ids_list, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}
