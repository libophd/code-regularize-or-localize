from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from jepa_lm.registry import Registry

DECODERS: Registry[Decoder] = Registry("decoder")


class Decoder(nn.Module, ABC):
    def __init__(self, d_proj: int, vocab_size: int) -> None:
        super().__init__()
        self.d_proj = d_proj
        self.vocab_size = vocab_size

    @abstractmethod
    def decode_logits(self, z: Tensor) -> Tensor: ...

    @abstractmethod
    def auxiliary_loss(self, z: Tensor, target_tokens: Tensor) -> Tensor: ...


@DECODERS.register("alignment_nn")
class AlignmentNNDecoder(Decoder):
    def __init__(self, d_proj: int, vocab_size: int, embed_table: nn.Embedding, temperature: float = 0.07) -> None:
        super().__init__(d_proj, vocab_size)
        self.embed_table = embed_table
        self.temperature = temperature

    def decode_logits(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def auxiliary_loss(self, z: Tensor, target_tokens: Tensor) -> Tensor:
        raise NotImplementedError


@DECODERS.register("ce_aux")
class CEAuxDecoder(Decoder):
    def __init__(self, d_proj: int, vocab_size: int) -> None:
        super().__init__(d_proj, vocab_size)
        self.head = nn.Linear(d_proj, vocab_size, bias=False)

    def decode_logits(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def auxiliary_loss(self, z: Tensor, target_tokens: Tensor) -> Tensor:
        raise NotImplementedError


@DECODERS.register("gmm")
class GMMDecoder(Decoder):
    def __init__(self, d_proj: int, vocab_size: int, n_components: int = 8) -> None:
        super().__init__(d_proj, vocab_size)
        self.n_components = n_components

    def decode_logits(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def auxiliary_loss(self, z: Tensor, target_tokens: Tensor) -> Tensor:
        raise NotImplementedError
