from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from jepa_lm.registry import Registry
from jepa_lm.training.sigreg import sigreg_per_position

KV_BRIDGES: Registry[KVBridge] = Registry("kv_bridge")


class KVBridge(nn.Module, ABC):
    def __init__(self, n_layers: int, d_kv: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.d_kv = d_kv

    def _check_layers(self, k_per_layer: list[Tensor], v_per_layer: list[Tensor]) -> None:
        if len(k_per_layer) != self.n_layers or len(v_per_layer) != self.n_layers:
            raise ValueError(
                f"{type(self).__name__}: expected {self.n_layers} layers, got "
                f"K={len(k_per_layer)}, V={len(v_per_layer)}"
            )

    @abstractmethod
    def auxiliary_loss(
        self,
        k_per_layer: list[Tensor],
        v_per_layer: list[Tensor],
        *,
        z: Tensor | None = None,
    ) -> Tensor: ...


@KV_BRIDGES.register("multi_layer_sigreg")
class MultiLayerSIGRegBridge(KVBridge):
    def __init__(
        self,
        n_layers: int,
        d_kv: int,
        lambda_kv: float = 0.01,
        m_projections: int = 1024,
    ) -> None:
        super().__init__(n_layers, d_kv)
        self.lambda_kv = lambda_kv
        self.m_projections = m_projections

    def auxiliary_loss(
        self,
        k_per_layer: list[Tensor],
        v_per_layer: list[Tensor],
        *,
        z: Tensor | None = None,
    ) -> Tensor:
        self._check_layers(k_per_layer, v_per_layer)
        terms = [sigreg_per_position(k, m=self.m_projections) for k in k_per_layer]
        terms += [sigreg_per_position(v, m=self.m_projections) for v in v_per_layer]
        return self.lambda_kv * torch.stack(terms).sum()


@KV_BRIDGES.register("per_layer_kv_predictors")
class PerLayerKVPredictorBridge(KVBridge):
    def __init__(self, n_layers: int, d_kv: int) -> None:
        super().__init__(n_layers, d_kv)
        self.k_predictors = nn.ModuleList([self._make_predictor() for _ in range(n_layers)])
        self.v_predictors = nn.ModuleList([self._make_predictor() for _ in range(n_layers)])

    def _make_predictor(self) -> nn.Module:
        return nn.Linear(self.d_kv, self.d_kv)

    def auxiliary_loss(
        self,
        k_per_layer: list[Tensor],
        v_per_layer: list[Tensor],
        *,
        z: Tensor | None = None,
    ) -> Tensor:
        self._check_layers(k_per_layer, v_per_layer)
        if k_per_layer[0].shape[1] < 2:
            raise ValueError(
                "PerLayerKVPredictorBridge needs T >= 2 for next-position "
                f"prediction; got T={k_per_layer[0].shape[1]}"
            )
        terms: list[Tensor] = []
        for layer in range(self.n_layers):
            k = k_per_layer[layer]
            terms.append(F.mse_loss(self.k_predictors[layer](k[:, :-1, :]), k[:, 1:, :]))
            v = v_per_layer[layer]
            terms.append(F.mse_loss(self.v_predictors[layer](v[:, :-1, :]), v[:, 1:, :]))
        return torch.stack(terms).sum()


@KV_BRIDGES.register("per_layer_kv_mlp_predictors")
class MLPPerLayerKVPredictorBridge(PerLayerKVPredictorBridge):
    def __init__(self, n_layers: int, d_kv: int, hidden_multiplier: int = 2) -> None:
        if hidden_multiplier < 1:
            raise ValueError(f"hidden_multiplier must be >= 1, got {hidden_multiplier}")
        self.hidden_multiplier = hidden_multiplier
        super().__init__(n_layers, d_kv)

    def _make_predictor(self) -> nn.Module:
        hidden = self.hidden_multiplier * self.d_kv
        return nn.Sequential(
            nn.Linear(self.d_kv, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.d_kv),
        )


@KV_BRIDGES.register("post_hoc_kv_decoders")
class PostHocKVDecoderBridge(KVBridge):
    def __init__(self, n_layers: int, d_kv: int, d_proj: int) -> None:
        super().__init__(n_layers, d_kv)
        self.d_proj = d_proj
        self.k_decoders = nn.ModuleList([nn.Linear(d_proj, d_kv) for _ in range(n_layers)])
        self.v_decoders = nn.ModuleList([nn.Linear(d_proj, d_kv) for _ in range(n_layers)])

    def auxiliary_loss(
        self,
        k_per_layer: list[Tensor],
        v_per_layer: list[Tensor],
        *,
        z: Tensor | None = None,
    ) -> Tensor:
        self._check_layers(k_per_layer, v_per_layer)
        if z is None:
            raise ValueError(
                "PostHocKVDecoderBridge.auxiliary_loss requires z (projected embeddings)"
            )
        terms: list[Tensor] = []
        for layer in range(self.n_layers):
            terms.append(F.mse_loss(self.k_decoders[layer](z), k_per_layer[layer]))
            terms.append(F.mse_loss(self.v_decoders[layer](z), v_per_layer[layer]))
        return torch.stack(terms).sum()
