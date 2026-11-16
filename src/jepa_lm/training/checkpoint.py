from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from jepa_lm.training.trainer import Trainer

_FORMAT_VERSION = 1


def save_checkpoint(trainer: Trainer, path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    blob: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "global_step": trainer.global_step,
        "config": trainer.config,
        "trunk": trainer.trunk.state_dict(),
        "head": trainer.head.state_dict(),
        "loss_module": trainer.loss_module.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "grad_scaler": (
            trainer.grad_scaler.state_dict() if trainer.grad_scaler is not None else None
        ),
        "rng": _capture_rng_state(),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | os.PathLike[str]) -> Trainer:
    path = Path(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return load_checkpoint_blob(blob)


def load_checkpoint_blob(blob: dict) -> Trainer:
    version = blob.get("format_version")
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format_version {version!r} != supported {_FORMAT_VERSION}; "
            f"this checkpoint was written by an incompatible code revision"
        )

    trainer = Trainer(blob["config"])
    trainer.trunk.load_state_dict(blob["trunk"])
    trainer.head.load_state_dict(blob["head"])
    trainer.loss_module.load_state_dict(blob["loss_module"])
    trainer.optimizer.load_state_dict(blob["optimizer"])
    trainer.scheduler.load_state_dict(blob["scheduler"])

    saved_scaler = blob["grad_scaler"]
    if saved_scaler is not None:
        if trainer.grad_scaler is None:
            raise ValueError(
                "checkpoint carries GradScaler state but the reconstructed Trainer "
                "has no scaler (config.amp_dtype is not float16) — checkpoint/config "
                "mismatch"
            )
        trainer.grad_scaler.load_state_dict(saved_scaler)

    trainer.global_step = blob["global_step"]
    _restore_rng_state(blob["rng"])
    return trainer


def _capture_rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(rng: dict[str, Any]) -> None:
    torch.set_rng_state(rng["torch_cpu"])

    cuda_states = rng.get("torch_cuda")
    if cuda_states is None:
        return
    if not torch.cuda.is_available():
        return
    if len(cuda_states) != torch.cuda.device_count():
        return
    torch.cuda.set_rng_state_all(cuda_states)
