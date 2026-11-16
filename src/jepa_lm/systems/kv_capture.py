from __future__ import annotations

from contextlib import AbstractContextManager

import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle


def attention_projections(trunk: nn.Module) -> list[tuple[nn.Linear, nn.Linear]]:
    model = getattr(trunk, "model", None)
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"attention_projections: cannot find `.model.layers` on "
            f"{type(trunk).__name__}; only HFLlamaTrunk-style trunks are supported."
        )
    pairs: list[tuple[nn.Linear, nn.Linear]] = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        k_proj = getattr(attn, "k_proj", None)
        v_proj = getattr(attn, "v_proj", None)
        if not isinstance(k_proj, nn.Linear) or not isinstance(v_proj, nn.Linear):
            raise TypeError(
                f"attention_projections: layer {i} has no k_proj/v_proj Linear "
                f"(got {type(k_proj).__name__}/{type(v_proj).__name__})."
            )
        pairs.append((k_proj, v_proj))
    if not pairs:
        raise TypeError("attention_projections: trunk has zero layers.")
    return pairs


def query_projections(trunk: nn.Module) -> list[nn.Linear]:
    model = getattr(trunk, "model", None)
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"query_projections: cannot find `.model.layers` on "
            f"{type(trunk).__name__}; only HFLlamaTrunk-style trunks are supported."
        )
    projs: list[nn.Linear] = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        q_proj = getattr(attn, "q_proj", None)
        if not isinstance(q_proj, nn.Linear):
            raise TypeError(
                f"query_projections: layer {i} has no q_proj Linear "
                f"(got {type(q_proj).__name__})."
            )
        projs.append(q_proj)
    if not projs:
        raise TypeError("query_projections: trunk has zero layers.")
    return projs


class KVCapture(AbstractContextManager):
    def __init__(self, trunk: nn.Module, *, capture_q: bool = False) -> None:
        self._pairs = attention_projections(trunk)
        self.n_layers = len(self._pairs)
        self.capture_q = capture_q
        self._q_projs = query_projections(trunk) if capture_q else []
        self.k_per_layer: list[Tensor | None] = [None] * self.n_layers
        self.v_per_layer: list[Tensor | None] = [None] * self.n_layers
        self.q_per_layer: list[Tensor | None] = [None] * self.n_layers
        self._handles: list[RemovableHandle] = []

    @staticmethod
    def _make_hook(idx: int, store: list[Tensor | None]):
        def hook(_module: nn.Module, _inputs: tuple, output: Tensor) -> None:
            store[idx] = output

        return hook

    def __enter__(self) -> "KVCapture":
        for i, (k_proj, v_proj) in enumerate(self._pairs):
            self._handles.append(k_proj.register_forward_hook(self._make_hook(i, self.k_per_layer)))
            self._handles.append(v_proj.register_forward_hook(self._make_hook(i, self.v_per_layer)))
        for i, q_proj in enumerate(self._q_projs):
            self._handles.append(q_proj.register_forward_hook(self._make_hook(i, self.q_per_layer)))
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        for store in (self.k_per_layer, self.v_per_layer, self.q_per_layer):
            for i in range(len(store)):
                store[i] = None

    def captured(self) -> tuple[list[Tensor], list[Tensor]]:
        if any(t is None for t in self.k_per_layer) or any(t is None for t in self.v_per_layer):
            missing = [i for i, t in enumerate(self.k_per_layer) if t is None]
            raise RuntimeError(
                f"KVCapture: no forward pass populated layers {missing or 'V'}; "
                "run the trunk forward inside the `with` block before calling captured()."
            )
        return self.k_per_layer, self.v_per_layer

    def captured_q(self) -> list[Tensor]:
        if not self.capture_q:
            raise RuntimeError(
                "KVCapture: capture_q=False; construct KVCapture(trunk, capture_q=True) to read Q."
            )
        if any(t is None for t in self.q_per_layer):
            missing = [i for i, t in enumerate(self.q_per_layer) if t is None]
            raise RuntimeError(
                f"KVCapture: no forward pass populated Q layers {missing}; "
                "run the trunk forward inside the `with` block before calling captured_q()."
            )
        return self.q_per_layer

    @property
    def d_kv(self) -> int:
        return self._pairs[0][0].out_features
