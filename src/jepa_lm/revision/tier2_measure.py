from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from jepa_lm.systems.bridge_probe import _repeat_kv, _split_heads
from jepa_lm.systems.kv_quant import fake_quantize
from jepa_lm.systems.rope import apply_rope
from jepa_lm.revision.sweep_configs import SWEEPS, label_for

PRIMARY_SEED = 42
LAYERWISE_BITS = 3
N_LAYERS = 12

ATTENTION_CELLS: tuple[dict[str, Any], ...] = (
    {"cell": "3bit-per_channel-k", "locus": "pre_rope",
     "bits": 3, "granularity": "per_channel", "mode": "symmetric",
     "group_size": None},
    {"cell": "3bit-per_channel-k-g32", "locus": "pre_rope",
     "bits": 3, "granularity": "per_channel", "mode": "symmetric",
     "group_size": 32},
    {"cell": "3bit-per_token-k", "locus": "pre_rope",
     "bits": 3, "granularity": "per_token", "mode": "symmetric",
     "group_size": None},
    {"cell": "3bit-per_channel-k-asym-g32", "locus": "pre_rope",
     "bits": 3, "granularity": "per_channel", "mode": "asymmetric",
     "group_size": 32},
    {"cell": "3bit-per_channel-k-asym-g32-postrope", "locus": "post_rope",
     "bits": 3, "granularity": "per_channel", "mode": "asymmetric",
     "group_size": 32},
)


def layerwise_matrix() -> list[dict[str, Any]]:
    out = []
    for granularity in ("per_token", "per_channel"):
        for target in ("k", "v"):
            for layer in range(N_LAYERS):
                out.append({
                    "layer": layer, "target": target,
                    "granularity": granularity, "bits": LAYERWISE_BITS,
                    "mode": "symmetric",
                    "config": f"{LAYERWISE_BITS}bit-{granularity}-{target}",
                })
    return out


def both_target_recon_cells() -> list[dict[str, Any]]:
    cells = []
    for sweep in ("bt3", "bt4"):
        for cfg in SWEEPS[sweep]():
            if cfg.get("bits") is None:
                continue
            if cfg.get("targets") != "both":
                raise ValueError(f"non-both-target cell in {sweep}: {cfg}")
            cells.append({**cfg, "config": label_for(cfg)})
    labels = [c["config"] for c in cells]
    if len(labels) != len(set(labels)) or len(labels) != 76:
        raise ValueError(f"recon cell inventory broken: {len(labels)} "
                         f"cells, {len(set(labels))} unique")
    return cells


def recon_quantize(x: Tensor, tensor: str, cfg: dict[str, Any]) -> Tensor:
    granularity = cfg["granularity"]
    if granularity == "mixed":
        granularity = "per_channel" if tensor == "k" else "per_token"
    return fake_quantize(x, cfg["bits"], granularity, cfg["mode"],
                         group_size=cfg.get("group_size"))


class ReconAccumulator:
    def __init__(self) -> None:
        self.num: dict[tuple[int, str, str], float] = {}
        self.den: dict[tuple[int, str], float] = {}
        self.batches: dict[tuple[int, str], int] = {}

    def update(self, layer: int, tensor: str, x: Tensor,
               cells: list[dict[str, Any]]) -> None:
        xd = x.double()
        dkey = (layer, tensor)
        self.den[dkey] = self.den.get(dkey, 0.0) + float((xd * xd).sum())
        self.batches[dkey] = self.batches.get(dkey, 0) + 1
        for cell in cells:
            xq = recon_quantize(x, tensor, cell)
            d = xd - xq.double()
            key = (layer, tensor, cell["config"])
            self.num[key] = self.num.get(key, 0.0) + float((d * d).sum())

    def value(self, layer: int, tensor: str, config: str) -> float:
        den = self.den[(layer, tensor)]
        if not math.isfinite(den) or den <= 0.0:
            raise ValueError(f"degenerate denominator for "
                             f"({layer}, {tensor}): {den}")
        return math.sqrt(self.num[(layer, tensor, config)] / den)


def quantize_k_heads(k_flat: Tensor, cell: dict[str, Any], *,
                     n_kv: int, positions: Tensor,
                     rope_base: float) -> Tensor:
    if cell["locus"] == "pre_rope":
        q = fake_quantize(k_flat, cell["bits"], cell["granularity"],
                          cell["mode"], group_size=cell["group_size"])
        return apply_rope(_split_heads(q, n_kv), positions, base=rope_base)
    if cell["locus"] != "post_rope":
        raise ValueError(f"unknown locus {cell['locus']!r}")
    heads = apply_rope(_split_heads(k_flat, n_kv), positions,
                       base=rope_base)
    b, h, t, hd = heads.shape
    flat = heads.transpose(1, 2).reshape(b, t, h * hd)
    quant = fake_quantize(flat, cell["bits"], cell["granularity"],
                          cell["mode"], group_size=cell["group_size"])
    return quant.reshape(b, t, h, hd).transpose(1, 2)


def attention_layer_sums(q_flat: Tensor, k_flat: Tensor,
                         cells: tuple[dict[str, Any], ...] = ATTENTION_CELLS,
                         *, n_heads: int, n_kv: int, rope_base: float,
                         head_chunk: int = 2) -> dict[str, dict[str, float]]:
    b, t, d = q_flat.shape
    positions = torch.arange(t, device=q_flat.device)
    n_rep = n_heads // n_kv
    qh = apply_rope(_split_heads(q_flat, n_heads), positions,
                    base=rope_base)
    k_true = _repeat_kv(
        apply_rope(_split_heads(k_flat, n_kv), positions, base=rope_base),
        n_rep)
    k_cells = {
        cell["cell"]: _repeat_kv(
            quantize_k_heads(k_flat, cell, n_kv=n_kv, positions=positions,
                             rope_base=rope_base), n_rep)
        for cell in cells
    }
    hd = d // n_heads
    scale = hd ** -0.5
    tril = torch.tril(torch.ones(t, t, dtype=torch.bool,
                                 device=q_flat.device))
    out = {name: {"logit_num": 0.0, "logit_den": 0.0,
                  "map_num": 0.0, "map_den": 0.0}
           for name in k_cells}
    for h0 in range(0, n_heads, head_chunk):
        qc = qh[:, h0:h0 + head_chunk].float()
        st = (qc @ k_true[:, h0:h0 + head_chunk].float().transpose(-1, -2)
              ) * scale
        st_tril = st[..., tril]
        mt = torch.softmax(
            st.masked_fill(~tril, float("-inf")), dim=-1)
        for name, kq in k_cells.items():
            sq = (qc @ kq[:, h0:h0 + head_chunk].float().transpose(-1, -2)
                  ) * scale
            dlog = (st_tril.double() - sq[..., tril].double())
            out[name]["logit_num"] += float((dlog * dlog).sum())
            ref = st_tril.double()
            out[name]["logit_den"] += float((ref * ref).sum())
            mq = torch.softmax(
                sq.masked_fill(~tril, float("-inf")), dim=-1)
            dmap = (mt.double() - mq.double())
            out[name]["map_num"] += float((dmap * dmap).sum())
            mref = mt.double()
            out[name]["map_den"] += float((mref * mref).sum())
    return out
