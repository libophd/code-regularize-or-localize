from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from jepa_lm.registry import Registry

LM_HEADS: Registry[LMHead] = Registry("lm_head")


class LMHead(nn.Module, ABC):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

    @abstractmethod
    def forward(self, h: Tensor) -> Tensor: ...


@LM_HEADS.register("linear")
class LinearLMHead(LMHead):
    def __init__(self, d_model: int, vocab_size: int, bias: bool = False) -> None:
        super().__init__(d_model, vocab_size)
        self.proj = nn.Linear(d_model, vocab_size, bias=bias)

    def tie_to(self, embedding: nn.Embedding) -> None:
        if embedding.num_embeddings != self.vocab_size:
            raise ValueError(
                f"tie_to: embedding.num_embeddings={embedding.num_embeddings} "
                f"!= vocab_size={self.vocab_size}"
            )
        if embedding.embedding_dim != self.d_model:
            raise ValueError(
                f"tie_to: embedding.embedding_dim={embedding.embedding_dim} "
                f"!= d_model={self.d_model}"
            )
        self.proj.weight = embedding.weight

    def forward(self, h: Tensor) -> Tensor:
        if h.dim() not in (2, 3):
            raise ValueError(f"LinearLMHead expects 2D or 3D input; got shape {tuple(h.shape)}")
        if h.shape[-1] != self.d_model:
            raise ValueError(
                f"LinearLMHead: input last dim {h.shape[-1]} != d_model={self.d_model}"
            )
        return F.linear(h, self.proj.weight, self.proj.bias)
