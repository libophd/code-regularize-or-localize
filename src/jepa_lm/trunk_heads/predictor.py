from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from jepa_lm.registry import Registry

PREDICTORS: Registry[PredictorP] = Registry("predictor")


class PredictorP(nn.Module, ABC):
    def __init__(self, d_model: int, d_proj: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_proj = d_proj

    @abstractmethod
    def forward(self, h: Tensor) -> Tensor: ...


@PREDICTORS.register("separate_head_mlp")
class SeparateHeadMLPPredictor(PredictorP):
    def __init__(self, d_model: int, d_proj: int, hidden_dim: int | None = None) -> None:
        super().__init__(d_model, d_proj)
        h = hidden_dim if hidden_dim is not None else 4 * d_proj
        self.hidden_dim = h
        self.net = nn.Sequential(
            nn.Linear(d_model, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, d_proj),
        )

    def forward(self, h: Tensor) -> Tensor:
        if h.dim() not in (2, 3):
            raise ValueError(f"SeparateHeadMLPPredictor expects 2D or 3D input; got shape {tuple(h.shape)}")
        return self.net(h)


@PREDICTORS.register("tied_weights_pred_tokens")
class TiedWeightsPredTokensPredictor(PredictorP):
    def __init__(self, d_model: int, d_proj: int, k_pred_tokens: int = 1) -> None:
        super().__init__(d_model, d_proj)
        self.k_pred_tokens = k_pred_tokens

    def forward(self, h: Tensor) -> Tensor:
        raise NotImplementedError(
            "TiedWeightsPredTokensPredictor is trunk-coupled; implement once the LM trunk is wired."
        )


@PREDICTORS.register("identity")
class IdentityPredictor(PredictorP):
    def __init__(self, d_model: int, d_proj: int | None = None) -> None:
        if d_proj is not None and d_proj != d_model:
            raise ValueError(
                f"IdentityPredictor requires d_proj == d_model; got d_model={d_model}, d_proj={d_proj}"
            )
        super().__init__(d_model, d_model)

    def forward(self, h: Tensor) -> Tensor:
        return h
