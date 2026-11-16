from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from jepa_lm.registry import Registry

TRUNKS: Registry[Trunk] = Registry("trunk")


class Trunk(nn.Module, ABC):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

    @abstractmethod
    def get_input_embeddings(self) -> nn.Embedding: ...

    @abstractmethod
    def forward(self, input_ids: Tensor) -> Tensor: ...


@TRUNKS.register("hf_llama")
class HFLlamaTrunk(Trunk):
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 2048,
        vocab_size: int = 32000,
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__(d_model=hidden_size, vocab_size=vocab_size)
        from transformers import LlamaConfig, LlamaModel

        config = LlamaConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            use_cache=False,
        )
        self.model = LlamaModel(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                f"HFLlamaTrunk expects (B, T) input_ids; got {tuple(input_ids.shape)}"
            )
        out = self.model(input_ids=input_ids, use_cache=False)
        return out.last_hidden_state
