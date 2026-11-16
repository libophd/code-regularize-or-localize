from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from jepa_lm.registry import Registry

ADAPTERS: Registry[Adapter] = Registry("adapter")


class Adapter(nn.Module, ABC):
    def __init__(self, d_proj: int, d_embed: int) -> None:
        super().__init__()
        self.d_proj = d_proj
        self.d_embed = d_embed

    @abstractmethod
    def forward(self, z_hat: Tensor) -> Tensor: ...


@ADAPTERS.register("a_mlp")
class AMLPAdapter(Adapter):
    def __init__(self, d_proj: int, d_embed: int, hidden_dim: int | None = None) -> None:
        super().__init__(d_proj, d_embed)
        self.hidden_dim = hidden_dim or d_proj

    def forward(self, z_hat: Tensor) -> Tensor:
        raise NotImplementedError


@ADAPTERS.register("a_topk_ann")
class ATopKANNAdapter(Adapter):
    def __init__(
        self,
        d_proj: int,
        d_embed: int,
        embed_table: nn.Embedding,
        k: int = 8,
        index_kind: str = "hnsw",
    ) -> None:
        super().__init__(d_proj, d_embed)
        self.embed_table = embed_table
        self.k = k
        self.index_kind = index_kind

    def build_index(self) -> None:
        raise NotImplementedError

    def forward(self, z_hat: Tensor) -> Tensor:
        raise NotImplementedError


@ADAPTERS.register("a_soft")
class ASoftAdapter(Adapter):
    def __init__(self, d_proj: int, d_embed: int, embed_table: nn.Embedding, temperature: float = 1.0) -> None:
        super().__init__(d_proj, d_embed)
        self.embed_table = embed_table
        self.temperature = temperature

    def forward(self, z_hat: Tensor) -> Tensor:
        raise NotImplementedError
