from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from jepa_lm.registry import Registry
from jepa_lm.training.sigreg import sigreg, sigreg_per_position

LOSS_TERMS: Registry[LossTerm] = Registry("loss_term")


class LossTerm(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: Mapping[str, Tensor]) -> Tensor: ...


@LOSS_TERMS.register("l_pred")
class PredLoss(LossTerm):
    REQUIRED_KEYS = ("p", "g")

    def __init__(self, stop_grad_target: bool = False) -> None:
        super().__init__()
        self.stop_grad_target = stop_grad_target

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        p = batch["p"]
        g = batch["g"]
        if p.dim() != 3 or g.dim() != 3:
            raise ValueError(
                f"PredLoss expects (B, T, D) for p and g; got p={tuple(p.shape)}, g={tuple(g.shape)}"
            )
        if p.shape != g.shape:
            raise ValueError(f"PredLoss: p shape {tuple(p.shape)} != g shape {tuple(g.shape)}")
        seq_len = p.shape[1]
        if seq_len < 2:
            raise ValueError(f"PredLoss requires T >= 2 for next-step pairs; got T={seq_len}")

        p_src = p[:, :-1]
        g_tgt = g[:, 1:]
        if self.stop_grad_target:
            g_tgt = g_tgt.detach()
        sq_err = (p_src - g_tgt).square()

        mask = batch.get("attention_mask")
        if mask is None:
            return sq_err.mean()
        if mask.dim() != 2 or mask.shape != (p.shape[0], seq_len):
            raise ValueError(
                f"PredLoss attention_mask must be (B, T)={(p.shape[0], seq_len)}; got {tuple(mask.shape)}"
            )
        pair_mask = (mask[:, :-1] * mask[:, 1:]).to(sq_err.dtype)
        d_proj = sq_err.shape[-1]
        denom = (pair_mask.sum() * d_proj).clamp_min(1.0)
        return (sq_err * pair_mask.unsqueeze(-1)).sum() / denom


@LOSS_TERMS.register("l_sigreg")
class SIGRegLoss(LossTerm):
    def __init__(self, m_projections: int = 1024, key: str = "g") -> None:
        super().__init__()
        if m_projections < 1:
            raise ValueError(f"m_projections must be >= 1; got {m_projections}")
        self.m_projections = m_projections
        self.key = key

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        z = batch[self.key]
        if z.dim() != 3:
            raise ValueError(
                f"SIGRegLoss expects (B, T, D) at batch[{self.key!r}]; got {tuple(z.shape)}"
            )
        if "attention_mask" in batch:
            raise ValueError(
                "SIGRegLoss (per-position) cannot honor batch['attention_mask']; "
                "pad timesteps would pollute the per-(timestep, projection) "
                "empirical CF. Use l_sigreg_pooled, or drop the mask key if the "
                "batch is genuinely dense."
            )
        return sigreg_per_position(z, m=self.m_projections)


@LOSS_TERMS.register("l_sigreg_pooled")
class SIGRegPooledLoss(LossTerm):
    def __init__(self, m_projections: int = 1024, key: str = "g") -> None:
        super().__init__()
        if m_projections < 1:
            raise ValueError(f"m_projections must be >= 1; got {m_projections}")
        self.m_projections = m_projections
        self.key = key

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        z = batch[self.key]
        if z.dim() != 3:
            raise ValueError(
                f"SIGRegPooledLoss expects (B, T, D) at batch[{self.key!r}]; got {tuple(z.shape)}"
            )
        b, length, d = z.shape
        z_flat = z.reshape(b * length, d)
        mask = batch.get("attention_mask")
        if mask is not None:
            if mask.dim() != 2 or mask.shape != (b, length):
                raise ValueError(
                    f"SIGRegPooledLoss attention_mask must be (B, T)={(b, length)}; got {tuple(mask.shape)}"
                )
            valid = mask.reshape(-1).bool()
            z_flat = z_flat[valid]
            if z_flat.shape[0] == 0:
                return z.sum() * 0.0
        return sigreg(z_flat, m=self.m_projections)


@LOSS_TERMS.register("l_aux")
class AuxCELoss(LossTerm):
    REQUIRED_KEYS = ("logits", "labels")

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        logits_key: str = "logits",
        labels_key: str = "labels",
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.logits_key = logits_key
        self.labels_key = labels_key

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        logits = batch[self.logits_key]
        labels = batch[self.labels_key]
        if logits.dim() != 3:
            raise ValueError(
                f"AuxCELoss expects (B, T, V) at batch[{self.logits_key!r}]; got {tuple(logits.shape)}"
            )
        if labels.dim() != 2 or labels.shape != logits.shape[:2]:
            raise ValueError(
                f"AuxCELoss labels must be (B, T)={tuple(logits.shape[:2])}; got {tuple(labels.shape)}"
            )
        seq_len = logits.shape[1]
        if seq_len < 2:
            raise ValueError(f"AuxCELoss requires T >= 2 for next-token shift; got T={seq_len}")

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab = shift_logits.shape[-1]

        valid = shift_labels != self.ignore_index
        mask = batch.get("attention_mask")
        if mask is not None:
            if mask.dim() != 2 or mask.shape != labels.shape:
                raise ValueError(
                    f"AuxCELoss attention_mask must be (B, T)={tuple(labels.shape)}; got {tuple(mask.shape)}"
                )
            pair_mask = (mask[:, :-1] != 0) & (mask[:, 1:] != 0)
            valid = valid & pair_mask

        safe_labels = torch.where(
            valid, shift_labels, torch.full_like(shift_labels, self.ignore_index)
        )
        per_token = F.cross_entropy(
            shift_logits.reshape(-1, vocab),
            safe_labels.reshape(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        valid_f = valid.reshape(-1).to(per_token.dtype)
        return (per_token * valid_f).sum() / valid_f.sum().clamp_min(1.0)


@LOSS_TERMS.register("l_align")
class AlignInfoNCELoss(LossTerm):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        raise NotImplementedError


@LOSS_TERMS.register("l_adapter")
class AdapterLoss(LossTerm):
    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        raise NotImplementedError


@LOSS_TERMS.register("l_stp")
class STPLoss(LossTerm):
    def __init__(self, max_span: int | None = None) -> None:
        super().__init__()
        self.max_span = max_span

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        raise NotImplementedError


@dataclass
class WeightedTerm:
    name: str
    weight: float


class JEPALMLoss(nn.Module):
    def __init__(self, terms: list[WeightedTerm], term_kwargs: Mapping[str, Mapping] | None = None) -> None:
        super().__init__()
        if not terms:
            raise ValueError("JEPALMLoss requires at least one term")
        term_kwargs = term_kwargs or {}
        self.terms = nn.ModuleDict(
            {wt.name: LOSS_TERMS.build(wt.name, **term_kwargs.get(wt.name, {})) for wt in terms}
        )
        self.weights = {wt.name: wt.weight for wt in terms}

    def forward(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        components: dict[str, Tensor] = {}
        total: Tensor | None = None
        for name, term in self.terms.items():
            value = term(batch)
            components[name] = value.detach()
            weighted = self.weights[name] * value
            total = weighted if total is None else total + weighted
        assert total is not None
        return total, components
