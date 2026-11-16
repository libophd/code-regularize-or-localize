from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from jepa_lm.token_io.lm_heads import LM_HEADS
from jepa_lm.training.losses import JEPALMLoss, WeightedTerm
from jepa_lm.trunks.llama import TRUNKS


@dataclass
class TrainerConfig:
    trunk_name: str = "hf_llama"
    trunk_kwargs: Mapping[str, Any] = field(default_factory=dict)
    lm_head_name: str = "linear"
    lm_head_kwargs: Mapping[str, Any] = field(default_factory=dict)
    tie_lm_head: bool = True

    loss_terms: list[WeightedTerm] = field(default_factory=list)
    loss_term_kwargs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    lr: float = 3e-4
    weight_decay: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8

    n_steps: int = 1000
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1

    grad_clip: float = 1.0
    log_every: int = 50
    seed: int = 42

    device: torch.device | None = None

    amp_dtype: torch.dtype | None = None


def _warmup_cosine_lambda(step: int, *, warmup_steps: int, n_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if step >= n_steps:
        return min_lr_ratio
    decay_total = max(1, n_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_total
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


class Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        if not config.loss_terms:
            raise ValueError("Trainer requires at least one loss term")
        if config.amp_dtype is not None and config.amp_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"amp_dtype must be None, torch.bfloat16, or torch.float16; got {config.amp_dtype!r}"
            )
        self.config = config
        torch.manual_seed(config.seed)

        self.trunk = TRUNKS.build(config.trunk_name, **config.trunk_kwargs)

        head_kwargs = dict(config.lm_head_kwargs)
        head_kwargs.update(d_model=self.trunk.d_model, vocab_size=self.trunk.vocab_size)
        self.head = LM_HEADS.build(config.lm_head_name, **head_kwargs)
        if config.tie_lm_head:
            self.head.tie_to(self.trunk.get_input_embeddings())

        self.loss_module = JEPALMLoss(
            terms=list(config.loss_terms),
            term_kwargs=config.loss_term_kwargs,
        )

        self.device = config.device or torch.device("cpu")
        self.trunk.to(self.device)
        self.head.to(self.device)
        self.loss_module.to(self.device)

        params = list({
            id(p): p
            for p in chain(self.trunk.parameters(), self.head.parameters())
        }.values())
        self.optimizer = AdamW(
            params,
            lr=config.lr,
            betas=config.adam_betas,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _warmup_cosine_lambda(
                step,
                warmup_steps=config.warmup_steps,
                n_steps=config.n_steps,
                min_lr_ratio=config.min_lr_ratio,
            ),
        )

        self.grad_scaler: torch.amp.GradScaler | None = None
        if config.amp_dtype is torch.float16:
            self.grad_scaler = torch.amp.GradScaler(device=self.device.type)

        self.global_step = 0

    def _enrich_batch(self, raw_batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        input_ids = raw_batch["input_ids"]
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        labels = raw_batch.get("labels", input_ids).to(self.device)

        h = self.trunk(input_ids)
        logits = self.head(h)
        return {"input_ids": input_ids, "labels": labels, "h": h, "logits": logits}

    def step(self, raw_batch: Mapping[str, Tensor]) -> dict[str, Any]:
        self.trunk.train()
        self.head.train()

        if self.config.amp_dtype is not None:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.config.amp_dtype):
                batch = self._enrich_batch(raw_batch)
                total, components = self.loss_module(batch)
        else:
            batch = self._enrich_batch(raw_batch)
            total, components = self.loss_module(batch)

        self.optimizer.zero_grad(set_to_none=True)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            total.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            (p for p in self.optimizer.param_groups[0]["params"]),
            self.config.grad_clip,
        )
        lr_used = self.optimizer.param_groups[0]["lr"]
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        lr_next = self.scheduler.get_last_lr()[0]

        return {
            "step": self.global_step,
            "loss": total.detach().item(),
            "components": {k: v.item() for k, v in components.items()},
            "lr": lr_used,
            "lr_next": lr_next,
            "grad_norm": float(grad_norm),
        }

    def train(
        self,
        dataloader: Iterable[Mapping[str, Tensor]],
        *,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = 0,
        on_step: Callable[[dict[str, Any]], None] | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        if checkpoint_every < 0:
            raise ValueError(f"checkpoint_every must be >= 0; got {checkpoint_every}")
        if checkpoint_every > 0 and checkpoint_dir is None:
            raise ValueError("checkpoint_every > 0 requires checkpoint_dir")

        last_metrics: dict[str, Any] = {}
        for raw_batch in dataloader:
            if self.global_step >= self.config.n_steps:
                break
            last_metrics = self.step(raw_batch)
            if on_step is not None:
                on_step(last_metrics)
            if verbose and self.global_step % self.config.log_every == 0:
                print(last_metrics)
            if checkpoint_every > 0 and self.global_step % checkpoint_every == 0:
                from jepa_lm.training.checkpoint import save_checkpoint

                save_checkpoint(
                    self,
                    Path(checkpoint_dir) / f"checkpoint_step{self.global_step}.pt",
                )
        return last_metrics
