from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import kv_quant_eval as kqe
from jepa_lm.data.fineweb import build_fineweb_factories
from jepa_lm.revision.envcheck import (
    ANCHOR_FREEZE_REL_PATH,
    ANCHOR_FREEZE_SHA256,
    check_freeze_complete,
    run_nvidia_smi,
)
from jepa_lm.revision.guards import (
    assert_lock_owned,
    claim_output_lock_isolated,
    enforce_single_visible_gpu,
    publish_no_replace_isolated,
    release_output_lock,
)
from jepa_lm.revision.hf_pins import (
    hf_cache_dir,
    load_pinned_fineweb,
    load_pinned_tokenizer,
    resolve_cached_revision,
)
from jepa_lm.revision.sweep_configs import (
    PINNED_N_BATCHES,
    PINNED_PROTOCOL,
    REGISTERED_MODEL_SHA256,
    assert_isolated_out_path,
    assert_legacy_trees_locked,
)

EVAL_SEED = PINNED_PROTOCOL["eval_seed"]


class CountingIter:
    def __init__(self, inner: Iterable[Mapping[str, torch.Tensor]]) -> None:
        self._inner, self.count = inner, 0

    def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]:
        for batch in self._inner:
            self.count += 1
            yield batch


def git_state() -> dict[str, Any]:
    import subprocess

    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(
                cmd, cwd=_REPO_ROOT, text=True).strip()
        except Exception as e:
            raise SystemExit(
                f"[tier2] git command {' '.join(cmd)} failed ({e}) — "
                "refusing to run with unknown code identity") from None

    sha = _run(["git", "rev-parse", "HEAD"])
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise SystemExit(f"[tier2] git sha {sha!r} is not 40-hex")
    dirty = _run(["git", "status", "--porcelain", "--",
                  ".", ":!docs"])
    return {"sha": sha, "dirty": bool(dirty)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registered_model(arm: str, seed: int,
                          device: torch.device) -> tuple[Any, Any, Path, str]:
    candidates = {ident: sha for sha, ident in
                  REGISTERED_MODEL_SHA256.items()}
    expected_sha = candidates.get((arm, seed))
    if expected_sha is None:
        raise SystemExit(f"[tier2] ({arm}, s{seed}) not in the registry")
    from jepa_lm.revision.pass1_tables import REGISTERED_MODEL_PATHS
    path = _REPO_ROOT / REGISTERED_MODEL_PATHS[(arm, seed)]
    got = sha256_file(path)
    if got != expected_sha:
        raise SystemExit(
            f"[tier2] {path} sha256 {got[:12]}... != registered "
            f"{expected_sha[:12]}... for ({arm}, s{seed})")
    import io
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise SystemExit(f"[tier2] {path} bytes changed between checks")
    blob = torch.load(io.BytesIO(data), map_location="cpu",
                      weights_only=False)
    del data
    trunk, head, _loader_kind = kqe.load_model_blob(blob)
    trunk.to(device).eval()
    head.to(device).eval()
    return trunk, head, path, got


def environment_gate(device: torch.device) -> dict[str, str]:
    enforce_single_visible_gpu(device)
    pkgs = check_freeze_complete(_REPO_ROOT,
                                 rel_path=ANCHOR_FREEZE_REL_PATH,
                                 expected_sha256=ANCHOR_FREEZE_SHA256)
    assert_legacy_trees_locked(_REPO_ROOT)
    return pkgs


def pinned_eval_factory() -> tuple[Callable[[], Any], int, dict[str, str]]:
    p = PINNED_PROTOCOL
    tokenizer_revision = resolve_cached_revision("models",
                                                 p["tokenizer_name"])
    dataset_revision = resolve_cached_revision("datasets",
                                               p["dataset_name"])
    tokenizer = load_pinned_tokenizer(p["tokenizer_name"],
                                      tokenizer_revision,
                                      expected_vocab=32000)
    n_eval_batches = math.ceil(p["eval_tokens"]
                               / (p["batch_size"] * p["seq_len"]))
    if n_eval_batches != PINNED_N_BATCHES:
        raise SystemExit("[tier2] pinned window arithmetic drifted")

    def raw_stream_factory():
        return load_pinned_fineweb(p["dataset_name"], p["dataset_config"],
                                   "train", dataset_revision)

    def tokenize_fn(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    _train, eval_factory = build_fineweb_factories(
        raw_stream_factory, tokenize_fn,
        seq_len=p["seq_len"], batch_size=p["batch_size"],
        eos_token_id=tokenizer.eos_token_id,
        eval_docs=p["eval_docs"], eval_offset_docs=p["eval_offset_docs"],
        eval_n_batches=n_eval_batches,
    )
    return eval_factory, n_eval_batches, {
        "tokenizer_repo": p["tokenizer_name"],
        "tokenizer_revision": tokenizer_revision,
        "dataset_repo": p["dataset_name"],
        "dataset_revision": dataset_revision,
    }


def env_record(device: torch.device) -> dict[str, Any]:
    from importlib.metadata import version

    def _v(pkg: str) -> str:
        try:
            return version(pkg)
        except Exception:
            return "unknown"

    return {
        "python": sys.version.split()[0],
        "torch": _v("torch"),
        "transformers": _v("transformers"),
        "datasets": _v("datasets"),
        "tokenizers": _v("tokenizers"),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "device_name": (torch.cuda.get_device_name(device)
                        if device.type == "cuda" else "cpu"),
        "visible_devices": (torch.cuda.device_count()
                            if device.type == "cuda" else 0),
        "hf_cache_dir": str(hf_cache_dir()),
        "nvidia_smi": run_nvidia_smi(),
    }


class IsolatedPublisher:
    def __init__(self, out_rel: str, required_root: str) -> None:
        self.out_path = _REPO_ROOT / out_rel
        self.required_root = required_root
        assert_isolated_out_path(self.out_path, _REPO_ROOT,
                                 required_root=required_root)
        if self.out_path.exists():
            raise SystemExit(f"[tier2] output exists: {self.out_path} — "
                             "registered outputs are never overwritten")
        self.git = git_state()
        if self.git["dirty"]:
            raise SystemExit("[tier2] dirty checkout — commit or stash "
                             "before a registered measurement run")
        assert_isolated_out_path(self.out_path, _REPO_ROOT,
                                 required_root=required_root)
        self.lock, self.token = claim_output_lock_isolated(_REPO_ROOT,
                                                           self.out_path)

    def release(self) -> None:
        release_output_lock(self.lock, self.token)

    def publish(self, envelope: dict[str, Any]) -> None:
        git_now = git_state()
        if git_now != self.git:
            self.release()
            raise SystemExit("[tier2] checkout changed mid-run — refusing "
                             "to publish")
        assert_lock_owned(self.lock, self.token)
        assert_isolated_out_path(self.out_path, _REPO_ROOT,
                                 required_root=self.required_root)
        publish_no_replace_isolated(
            _REPO_ROOT, self.out_path,
            json.dumps(envelope, indent=2, allow_nan=False) + "\n")
        self.release()


def finish() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
