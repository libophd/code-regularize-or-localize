from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from jepa_lm.registry import Registry

PROJECTORS: Registry[ProjectorG] = Registry("projector")


class ProjectorG(nn.Module, ABC):
    def __init__(self, d_model: int, d_proj: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_proj = d_proj

    @abstractmethod
    def forward(self, h: Tensor) -> Tensor: ...


@PROJECTORS.register("batchnorm_mlp")
class BatchNormMLPProjector(ProjectorG):
    def __init__(self, d_model: int, d_proj: int, hidden_dim: int | None = None) -> None:
        super().__init__(d_model, d_proj)
        h = hidden_dim if hidden_dim is not None else 4 * d_proj
        self.hidden_dim = h
        self.net = nn.Sequential(
            nn.Linear(d_model, h),
            nn.BatchNorm1d(h),
            nn.GELU(),
            nn.Linear(h, d_proj),
        )

    def forward(self, h: Tensor) -> Tensor:
        if h.dim() == 3:
            b, length, _ = h.shape
            return self.net(h.reshape(b * length, self.d_model)).reshape(b, length, self.d_proj)
        if h.dim() == 2:
            return self.net(h)
        raise ValueError(f"BatchNormMLPProjector expects 2D or 3D input; got shape {tuple(h.shape)}")


@PROJECTORS.register("layernorm_mlp")
class LayerNormMLPProjector(ProjectorG):
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
            raise ValueError(f"LayerNormMLPProjector expects 2D or 3D input; got shape {tuple(h.shape)}")
        return self.net(h)


@PROJECTORS.register("identity")
class IdentityProjector(ProjectorG):
    def __init__(self, d_model: int, d_proj: int | None = None) -> None:
        if d_proj is not None and d_proj != d_model:
            raise ValueError(
                f"IdentityProjector requires d_proj == d_model; got d_model={d_model}, d_proj={d_proj}"
            )
        super().__init__(d_model, d_model)

    def forward(self, h: Tensor) -> Tensor:
        return h
