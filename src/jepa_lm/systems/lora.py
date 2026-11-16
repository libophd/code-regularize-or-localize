from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LoraSpec:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"LoraSpec.rank must be >= 1; got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"LoraSpec.alpha must be > 0; got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"LoraSpec.dropout must be in [0, 1); got {self.dropout}")
        if not self.targets:
            raise ValueError("LoraSpec.targets must name at least one projection")

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["targets"] = list(self.targets)
        d["scaling"] = self.scaling
        return d


LORA_KV_SPEC = LoraSpec()


class LoRALinear(nn.Linear):
    def __init__(self, base: nn.Linear, spec: LoraSpec) -> None:
        if isinstance(base, LoRALinear):
            raise TypeError("LoRALinear: refusing to wrap an already-adapted layer")
        if type(base) is not nn.Linear:
            raise TypeError(f"LoRALinear: expected a plain nn.Linear base; got {type(base).__name__}")
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None,
                         device=base.weight.device, dtype=base.weight.dtype)
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.spec = spec
        self.scaling = spec.scaling
        self.lora_A = nn.Parameter(torch.empty(spec.rank, base.in_features,
                                               device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, spec.rank,
                                               device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(spec.dropout) if spec.dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(x, self.weight, self.bias)
        low_rank = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base + low_rank * self.scaling

    def lora_parameters(self) -> list[nn.Parameter]:
        return [self.lora_A, self.lora_B]

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        out = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None,
                        device=self.weight.device, dtype=self.weight.dtype)
        out.weight.copy_((self.weight.float() + delta).to(self.weight.dtype))
        if self.bias is not None:
            out.bias.copy_(self.bias)
        out.weight.requires_grad_(False)
        if out.bias is not None:
            out.bias.requires_grad_(False)
        return out


def _attention_modules(trunk: nn.Module) -> list[nn.Module]:
    model = getattr(trunk, "model", None)
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"lora: cannot find `.model.layers` on {type(trunk).__name__}; "
            "only HFLlamaTrunk-style trunks are supported."
        )
    attns = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise TypeError(f"lora: layer {i} has no `self_attn`")
        attns.append(attn)
    return attns


def inject_lora(trunk: nn.Module, spec: LoraSpec = LORA_KV_SPEC) -> list[nn.Parameter]:
    if lora_modules(trunk):
        raise RuntimeError("inject_lora: trunk already carries LoRA adapters")
    plan: list[tuple[nn.Module, str, nn.Linear]] = []
    for i, attn in enumerate(_attention_modules(trunk)):
        for name in spec.targets:
            base = getattr(attn, name, None)
            if type(base) is not nn.Linear:
                raise TypeError(
                    f"inject_lora: layer {i} `self_attn.{name}` is "
                    f"{type(base).__name__}, expected a plain nn.Linear"
                )
            plan.append((attn, name, base))
    for p in trunk.parameters():
        p.requires_grad_(False)
    trainable: list[nn.Parameter] = []
    for attn, name, base in plan:
        adapted = LoRALinear(base, spec)
        setattr(attn, name, adapted)
        trainable.extend(adapted.lora_parameters())
    return trainable


def lora_modules(trunk: nn.Module) -> list[tuple[str, LoRALinear]]:
    return [(n, m) for n, m in trunk.named_modules() if isinstance(m, LoRALinear)]


def lora_parameters(trunk: nn.Module) -> list[nn.Parameter]:
    return [p for _, m in lora_modules(trunk) for p in m.lora_parameters()]


def count_parameters(params) -> int:
    return int(sum(p.numel() for p in params))


def lora_state_dict(trunk: nn.Module) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, m in lora_modules(trunk):
        out[f"{name}.lora_A"] = m.lora_A.detach().cpu().clone()
        out[f"{name}.lora_B"] = m.lora_B.detach().cpu().clone()
    return out


@torch.no_grad()
def merge_lora(trunk: nn.Module) -> int:
    targets = lora_modules(trunk)
    for name, module in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = trunk.get_submodule(parent_name) if parent_name else trunk
        setattr(parent, attr, module.merged())
    return len(targets)
