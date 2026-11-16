from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch.nn as nn

from jepa_lm.systems.lora import LORA_KV_SPEC, LoraSpec

PROBE_NAME = "lora_kv"
PROBE_OUT_ROOT = "outputs/revision/probes"
PROBE_SEED = 42

LORA_PROBE_BUNDLE_FILES: tuple[str, ...] = (
    "config.json", "probe_result.json",
    "kv_quant_default.json", "kv_quant_kivi.json", "kv_quant_controls.json",
)
LORA_PROBE_BUNDLE_FILE_COUNT = len(LORA_PROBE_BUNDLE_FILES)


def lora_probe_bundle_sources() -> list[tuple[str, str]]:
    return [(f"probes/{PROBE_NAME}/s{PROBE_SEED}/{name}",
             f"{PROBE_OUT_ROOT}/{PROBE_NAME}/s{PROBE_SEED}/{name}")
            for name in LORA_PROBE_BUNDLE_FILES]


def lora_probe_bundle_inventory() -> list[str]:
    return [bundle for bundle, _ in lora_probe_bundle_sources()]


LORA_TRUNK_SHA256 = "4a2c28ea377f7b69c49ae92ee3bcb18bb77d31734b4f85f7a6338c01c862a151"
LORA_ADAPTER_SHA256 = "664f21eba7bbcbae81ca4128dc7772dcab3aed0120863dcd6de44dbe1bc13e10"
LORA_SOURCE_CHECKPOINT = ("outputs/phase1_seedconfirm_s42/checkpoints/arm1/sigreg_l0.01/"
                          "checkpoint_step610352.pt")
LORA_GRID_FILES: tuple[str, ...] = ("kv_quant_default.json", "kv_quant_kivi.json",
                                    "kv_quant_controls.json")


class LoraProbeAuthError(ValueError):
    pass


def verify_merge_reconstruction(merged_state: Mapping[str, Any],
                                source_state: Mapping[str, Any],
                                adapter_state: Mapping[str, Any],
                                spec: LoraSpec, *, atol: float = 1e-5) -> dict[str, int]:
    import torch

    merged_state = {k: v.detach().cpu() for k, v in merged_state.items()}
    source_state = {k: v.detach().cpu() for k, v in source_state.items()}
    adapter_state = {k: v.detach().cpu() for k, v in adapter_state.items()}
    if set(merged_state) != set(source_state):
        raise LoraProbeAuthError("merged/source trunk key sets differ")
    modules = sorted({k[: -len(".lora_A")] for k in adapter_state if k.endswith(".lora_A")})
    if not modules or any(f"{m}.lora_B" not in adapter_state for m in modules):
        raise LoraProbeAuthError("adapter state is missing lora_A/lora_B pairs")
    adapted_keys = {f"{m}.weight" for m in modules}
    if not adapted_keys <= set(merged_state):
        raise LoraProbeAuthError("adapter names a module the trunk does not have")
    for m in modules:
        a, b = adapter_state[f"{m}.lora_A"].float(), adapter_state[f"{m}.lora_B"].float()
        if a.shape[0] != spec.rank or b.shape[1] != spec.rank:
            raise LoraProbeAuthError(f"{m}: adapter rank {a.shape[0]}/{b.shape[1]} != {spec.rank}")
        expected = source_state[f"{m}.weight"].float() + (b @ a) * spec.scaling
        got = merged_state[f"{m}.weight"].float()
        if got.shape != expected.shape or not torch.allclose(got, expected, atol=atol, rtol=0.0):
            raise LoraProbeAuthError(f"{m}.weight is not source + adapter (max |diff| "
                                     f"{(got - expected).abs().max().item():.3e})")
    n_same = 0
    for k, v in merged_state.items():
        if k in adapted_keys:
            continue
        if not torch.equal(v, source_state[k]):
            raise LoraProbeAuthError(f"non-adapted tensor {k} differs from the source trunk")
        n_same += 1
    return {"adapted_modules": len(modules), "unchanged_tensors": n_same}


def authenticate_lora_probe(probe_dir: Path, *, source_trunk_state: Mapping[str, Any],
                            trunk_sha256: str = LORA_TRUNK_SHA256,
                            adapter_sha256: str = LORA_ADAPTER_SHA256,
                            expected_source_sha256: str | None = None,
                            source_head_state: Mapping[str, Any] | None = None,
                            expected_adapted_modules: int | None = None) -> dict[str, Any]:
    import hashlib
    import io
    import json

    import torch

    probe_dir = Path(probe_dir)
    trunk_bytes = (probe_dir / "adapted_trunk.pt").read_bytes()
    digest = hashlib.sha256(trunk_bytes).hexdigest()
    if digest != trunk_sha256:
        raise LoraProbeAuthError(f"adapted_trunk.pt digest {digest[:16]}… != frozen {trunk_sha256[:16]}…")
    blob = torch.load(io.BytesIO(trunk_bytes), map_location="cpu", weights_only=False)
    if (blob.get("format") != "a1_adapted_trunk_v1" or blob.get("arm") != PROBE_NAME
            or blob.get("merged") is not True or blob.get("dev_mode")):
        raise LoraProbeAuthError("trunk blob is not a merged, non-dev lora_kv blob")
    if any(".lora_A" in k or ".lora_B" in k for k in blob["trunk"]):
        raise LoraProbeAuthError("trunk blob still carries adapter tensors")
    sidecar = json.loads((probe_dir / "probe_result.json").read_text())
    if sidecar != blob["probe_result"]:
        raise LoraProbeAuthError("probe_result.json does not equal the blob's embedded probe_result")
    config = json.loads((probe_dir / "config.json").read_text())
    if config.get("recipe") != blob.get("recipe"):
        raise LoraProbeAuthError("config.json recipe does not equal the blob's embedded recipe")
    if config.get("dev_mode") or blob["recipe"].get("dev_mode"):
        raise LoraProbeAuthError("probe is tagged dev_mode")
    if expected_source_sha256 is not None and \
            blob["recipe"].get("source_checkpoint_sha256") != expected_source_sha256:
        raise LoraProbeAuthError("blob's source checkpoint digest is not the expected source")
    adapter_bytes = (probe_dir / "lora_adapter.pt").read_bytes()
    adigest = hashlib.sha256(adapter_bytes).hexdigest()
    if adigest != adapter_sha256:
        raise LoraProbeAuthError(f"lora_adapter.pt digest {adigest[:16]}… != frozen {adapter_sha256[:16]}…")
    adapter = torch.load(io.BytesIO(adapter_bytes), map_location="cpu", weights_only=False)
    if adapter.get("format") != "lora_adapter_v1" or adapter.get("steps") != sidecar.get("n_steps_trained"):
        raise LoraProbeAuthError("adapter file is not the run's lora_adapter_v1 at the trained step count")
    spec = spec_from_recipe(blob["recipe"])
    counts = verify_merge_reconstruction(blob["trunk"], source_trunk_state, adapter["state_dict"], spec)
    if expected_adapted_modules is not None and counts["adapted_modules"] != expected_adapted_modules:
        raise LoraProbeAuthError(f"{counts['adapted_modules']} adapted modules, expected {expected_adapted_modules}")
    if source_head_state is not None:
        head = {k: v.detach().cpu() for k, v in blob["head"].items()}
        src_head = {k: v.detach().cpu() for k, v in source_head_state.items()}
        if set(head) != set(src_head) or any(not torch.equal(head[k], src_head[k]) for k in head):
            raise LoraProbeAuthError("saved head does not equal the source checkpoint's head")
    trunk_rel = f"{PROBE_OUT_ROOT}/{PROBE_NAME}/s{config.get('source_seed', PROBE_SEED)}/adapted_trunk.pt"
    grids: dict[str, list] = {}
    for name in LORA_GRID_FILES:
        rows = json.loads((probe_dir / name).read_text())
        if not rows or any(r.get("model") != trunk_rel for r in rows):
            raise LoraProbeAuthError(f"{name}: a row does not name the authenticated trunk {trunk_rel}")
        grids[name] = rows
    return {"sidecar": sidecar, "config": config, "grids": grids,
            "trunk_sha256": digest, "adapter_sha256": adigest, **counts}

LORA_KV_RECIPE: dict[str, Any] = {
    "lr": 1e-4,
    "lambda_kv": 0.01,
    "m_projections": 256,
    "amp_dtype": "bf16",
    "unfreeze": "lora",
    "batch_size": 8,
    "seq_len": 2048,
    "train_tokens": 500_000_000,
    "eval_tokens": 2_000_000,
    "eval_docs": 5000,
    "eval_offset_docs": 100_000,
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_config": "sample-10BT",
    "dataset_split": "train",
    "tokenizer_name": "mistralai/Mistral-7B-v0.1",
    "vocab_size": 32000,
    "aniso_seed": 1234,
    "continuation_seed": 42,
    "source_seed": 42,
    "lora": LORA_KV_SPEC.to_dict(),
}

CODE_IDENTITY_PATHS: tuple[str, ...] = (
    "src",
    "scripts/revision/run_lora_probe.py",
    "scripts/kv_quant_eval.py",
)


def spec_from_recipe(recipe: Mapping[str, Any]) -> LoraSpec:
    lora = recipe["lora"]
    return LoraSpec(rank=int(lora["rank"]), alpha=float(lora["alpha"]),
                    dropout=float(lora["dropout"]), targets=tuple(lora["targets"]))


def recipe_deviations(recipe: Mapping[str, Any],
                      frozen: Mapping[str, Any] = LORA_KV_RECIPE) -> dict[str, tuple[Any, Any]]:
    return {k: (recipe.get(k), v) for k, v in frozen.items() if recipe.get(k) != v}


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise SystemExit(f"[lora_kv] git {' '.join(args)} failed ({e}) — "
                         "refusing to run with unknown code identity") from None


def code_identity(repo_root: Path) -> dict[str, Any]:
    sha = _git(repo_root, "rev-parse", "HEAD")
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise SystemExit(f"[lora_kv] git sha {sha!r} is not 40-hex — aborting")
    dirty = bool(_git(repo_root, "status", "--porcelain", "--", *CODE_IDENTITY_PATHS))
    dirty_anywhere = bool(_git(repo_root, "status", "--porcelain"))
    return {"sha": sha, "code_dirty": dirty, "tree_dirty": dirty_anywhere,
            "code_paths": list(CODE_IDENTITY_PATHS)}


def code_identity_unchanged(repo_root: Path, start: Mapping[str, Any]) -> tuple[bool, str]:
    now = code_identity(repo_root)
    if now["code_dirty"] != start["code_dirty"]:
        return False, f"code dirty state changed ({start['code_dirty']} -> {now['code_dirty']})"
    if now["sha"] != start["sha"]:
        changed = _git(repo_root, "diff", "--name-only", start["sha"], now["sha"], "--",
                       *CODE_IDENTITY_PATHS)
        if changed:
            return False, f"commits since {start['sha'][:12]} touched code paths: {changed.split()}"
    return True, "unchanged"


def build_lora_probe_blob(
    trunk: nn.Module,
    head: nn.Module,
    *,
    trunk_kwargs: Mapping[str, Any],
    tie_lm_head: bool,
    source_checkpoint: str,
    probe_result: Mapping[str, Any],
    recipe: Mapping[str, Any],
    dev_mode: bool = False,
) -> dict[str, Any]:
    keys = list(trunk.state_dict().keys())
    stray = [k for k in keys if ".lora_A" in k or ".lora_B" in k]
    if stray:
        raise ValueError(f"build_lora_probe_blob: trunk still carries adapters ({stray[:3]}…); "
                         "merge_lora before saving")
    return {
        "format": "a1_adapted_trunk_v1",
        "trunk": trunk.state_dict(),
        "head": head.state_dict(),
        "trunk_kwargs": dict(trunk_kwargs),
        "tie_lm_head": tie_lm_head,
        "source_checkpoint": source_checkpoint,
        "probe_result": dict(probe_result),
        "arm": PROBE_NAME,
        "probe": PROBE_NAME,
        "source_seed": int(recipe["source_seed"]),
        "continuation_seed": int(recipe["continuation_seed"]),
        "recipe": dict(recipe),
        "lora": dict(recipe["lora"]),
        "merged": True,
        "dev_mode": bool(dev_mode),
    }
