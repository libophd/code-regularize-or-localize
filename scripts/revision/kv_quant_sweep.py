from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import torch

import kv_quant_eval as kqe
from jepa_lm.data.fineweb import build_fineweb_factories
from jepa_lm.eval.evaluator import evaluate
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
from jepa_lm.revision.envcheck import (
    ANCHOR_FREEZE_REL_PATH,
    ANCHOR_FREEZE_SHA256,
    check_freeze_complete,
    run_nvidia_smi,
)
from jepa_lm.revision.sweep_configs import (
    APPROVED_TRAINING_COMMITS,
    CANONICAL_PILOT_LAUNCH,
    CANONICAL_PILOT_MANIFEST_OUT,
    REGISTERED_SWEEP_OUT_ROOT,
    assert_isolated_out_path,
    assert_legacy_trees_locked,
    validate_manifest_evidence,
    REGISTERED_ARMS as KNOWN_ARMS,
    RECIPE_PRESENCE_FIELDS,
    REGISTERED_MODEL_SHA256,
    SWEEPS,
    assert_pinned_protocol,
    frozen_new_arm_recipes,
    registered_arm1_sha256,
    strict_json_loads,
    validate_sweep_rows,
)

FROZEN_NEW_ARM_RECIPES = frozen_new_arm_recipes()
LEGACY_SIDECAR_LAMBDA = {"a1_adapted": 0.01, "ce_continue": 0.0}
LEGACY_GATE_SEMANTICS = {
    "a1_adapted": lambda pr: pr.get("passed") is True and pr.get("gate_value", 0) >= 0.30,
    "ce_continue": lambda pr: pr.get("passed") is False and pr.get("gate_value", 1) < 0.30,
}
LEGACY_BLOB_SHA256 = {
    "6a6e81e8e2687dda050f71f55083b2dcb744c7bfbdfb2ae4b7c4b32048c5de47": ("a1_adapted", 42),
    "292d3c3c0063a7dd8101a919b6ad53b0005454e3d060a372b6084203a009daf8": ("ce_continue", 42),
    "7f3a673ed9798ad04957b6bcfde51fa89f7cab3d2310f6aa2a694afb8505ab56": ("a1_adapted", 43),
    "5d0f53daa2d02ec1f8673234628285144386f7ed9555a5dbec0f21ad7c2e016a": ("ce_continue", 43),
    "e78af4c762052f0d07a39524ba2981e207ac10dfa2ff7ae8d9b577677ae46280": ("a1_adapted", 44),
    "064ea198a3e8c7237cd627f2403fff65117f53cce914c58e3a3dac587e8c7d6c": ("ce_continue", 44),
}


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_launch_manifest(
    manifest: Mapping[str, Any], *, manifest_path: str, model_path: str,
    model_sha256: str, arm: str, source_seed: int,
    continuation_seed: int | None,
    sweep: str, out_json: str, head_sha: str, head_dirty: bool,
    dev_dirty_ok: bool = False,
    expected_packages: Mapping[str, str] | None = None,
) -> None:
    if manifest.get("schema_version") != "launch-manifest-v1":
        raise ValueError(f"manifest schema {manifest.get('schema_version')!r} unknown")
    git = manifest.get("git") or {}
    if git.get("sha") != head_sha:
        raise ValueError(
            f"manifest commit {git.get('sha')!r} != current HEAD {head_sha!r}"
        )
    if not dev_dirty_ok and (git.get("dirty") is not False or head_dirty):
        raise ValueError("manifest and current checkout must both be clean")
    launch = manifest.get("launch") or {}
    want = {"model": model_path, "arm": arm, "source_seed": source_seed,
            "continuation_seed": continuation_seed,
            "sweep": sweep, "out_json": out_json}
    got = {k: launch.get(k) for k in want}
    if got != want:
        raise ValueError(f"manifest authorizes {got}, this run is {want}")
    entries = {m.get("path"): m.get("sha256") for m in manifest.get("models", [])}
    if entries.get(model_path) != model_sha256:
        raise ValueError(
            f"manifest model digest {entries.get(model_path)!r} != actual {model_sha256}"
        )
    if expected_packages is None:
        from importlib.metadata import version as _pkg_version

        expected_packages = {}
        for core in ("torch", "transformers", "datasets", "tokenizers"):
            try:
                expected_packages[core] = _pkg_version(core)
            except Exception:
                pass
    try:
        validate_manifest_evidence(manifest, allow_dirty=dev_dirty_ok,
                                   expected_packages=expected_packages)
    except AssertionError as e:
        raise ValueError(str(e)) from None
    if manifest.get("mode") == "pilot-bootstrap" \
            and manifest_path != CANONICAL_PILOT_MANIFEST_OUT:
        raise ValueError(
            f"bootstrap launch consumed manifest {manifest_path!r}, not the "
            f"canonical single-shot path {CANONICAL_PILOT_MANIFEST_OUT!r}")
    if manifest.get("mode") == "receipt" and (arm, source_seed, sweep) == (
            CANONICAL_PILOT_LAUNCH["arm"],
            CANONICAL_PILOT_LAUNCH["source_seed"],
            CANONICAL_PILOT_LAUNCH["sweep"]):
        raise ValueError(
            "the canonical pilot identity (l_aux, 42, st3) runs ONLY as the "
            "pilot-bootstrap artifact — receipt-mode relabeling is refused")


class _CountingIter:
    def __init__(self, inner: Iterable[Mapping[str, torch.Tensor]]) -> None:
        self._inner, self.count = inner, 0

    def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]:
        for batch in self._inner:
            self.count += 1
            yield batch


def verify_model_provenance(
    path: Path, *, arm: str, source_seed: int, continuation_seed: int | None,
    legacy_allowlist: Mapping[str, tuple[str, int]] | None = None,
    model_registry: Mapping[str, tuple[str, int]] | None = None,
    model_sha256: str | None = None,
    approved_training_commits: frozenset[str] | None = None,
) -> None:
    import hashlib as _hl

    registry = REGISTERED_MODEL_SHA256 if model_registry is None else model_registry
    allowlist = LEGACY_BLOB_SHA256 if legacy_allowlist is None else legacy_allowlist
    approved = (APPROVED_TRAINING_COMMITS if approved_training_commits is None
                else approved_training_commits)
    f = open(path, "rb")
    h = _hl.sha256()
    for chunk in iter(lambda: f.read(1 << 22), b""):
        h.update(chunk)
    held_digest = h.hexdigest()
    if model_sha256 is not None and held_digest != model_sha256:
        f.close()
        raise ValueError(
            f"model file changed between the caller's hash ({model_sha256[:16]}…) "
            f"and verification ({held_digest[:16]}…) — refusing"
        )
    model_sha256 = held_digest
    _new_arms = ("ce_only", "h_k_only", "h_v_only")
    if arm in _new_arms:
        if not approved:
            f.close()
            raise ValueError(
                "no APPROVED_TRAINING_COMMITS — new-arm blobs cannot be "
                "deserialized, let alone enter registered evaluation (populate "
                "APPROVED_TRAINING_COMMITS after the Tier-1 launch commit)"
            )
    else:
        known = registry.get(held_digest)
        if known is None and arm in ("a1_adapted", "ce_continue"):
            known = allowlist.get(held_digest)
        if known != (arm, source_seed):
            f.close()
            raise ValueError(
                f"model digest {held_digest[:16]}… is known as {known}, "
                f"claimed ({arm!r}, {source_seed}) — refusing BEFORE "
                "deserialization (unpickling an unauthenticated file would "
                "execute it)"
            )
    f.seek(0)
    try:
        blob = torch.load(f, map_location="cpu", weights_only=False)
    finally:
        f.close()
    if isinstance(blob, dict) and blob.get("format") == "a1_adapted_trunk_v1":
        if blob.get("dev_mode"):
            raise ValueError(
                "blob was trained in DEV MODE (--allow-off-recipe) — dev artifacts "
                "never enter registered evaluation"
            )

        if continuation_seed != 42:
            raise ValueError(
                f"adapted/continuation arms use continuation seed 42 by protocol; "
                f"got {continuation_seed}"
            )
        pr = blob.get("probe_result") or {}
        status = pr.get("status")
        if status is not None:
            raise ValueError(
                f"blob is a pre-eval snapshot (status={status!r}), not a completed run"
            )
        steps = pr.get("n_steps_trained")
        if steps is None:
            raise ValueError(
                "blob carries no n_steps_trained — completion evidence is REQUIRED"
            )
        if int(steps) != 30518:
            raise ValueError(f"blob records n_steps_trained={steps}, expected 30518")
        embedded_arm = blob.get("arm")
        if embedded_arm is not None:
            if embedded_arm != arm:
                raise ValueError(f"blob embeds arm={embedded_arm!r}, claimed {arm!r}")
            if int(blob.get("source_seed", -1)) != int(source_seed):
                raise ValueError(
                    f"blob embeds source_seed={blob.get('source_seed')}, claimed {source_seed}"
                )
            if int(blob.get("continuation_seed", -1)) != int(continuation_seed):
                raise ValueError(
                    f"blob embeds continuation_seed={blob.get('continuation_seed')}, "
                    f"claimed {continuation_seed}"
                )
            if embedded_arm not in FROZEN_NEW_ARM_RECIPES:
                raise ValueError(
                    f"blob embeds arm={embedded_arm!r}, which is not a "
                    "revision-era arm — legacy artifacts never embed an arm"
                )
            if blob.get("dev_mode") is not False:
                raise ValueError(
                    f"blob dev_mode={blob.get('dev_mode')!r} — must be "
                    "explicitly False for registered evaluation"
                )
            recipe = dict(blob.get("recipe") or {})
            if not recipe:
                raise ValueError(
                    f"revision-era blob for {embedded_arm!r} embeds no recipe — "
                    "off-recipe runs cannot be verified"
                )
            rdev = recipe.get("dev_mode")
            if rdev is not False:
                raise ValueError(
                    f"blob recipe.dev_mode={rdev!r} — must be explicitly False "
                    "and equal to the top-level flag"
                )
            expected_recipe = FROZEN_NEW_ARM_RECIPES[embedded_arm]
            pinned_view = {k: recipe.get(k) for k in expected_recipe}
            if pinned_view != expected_recipe:
                raise ValueError(
                    f"blob recipe pinned fields {pinned_view} != frozen {expected_recipe}"
                )
            missing = [k for k in RECIPE_PRESENCE_FIELDS if not recipe.get(k)]
            if missing:
                raise ValueError(f"blob recipe missing required fields {missing}")
            sha = str(recipe.get("git_sha", ""))
            if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError(f"blob recipe git_sha {sha!r} is not a 40-hex commit")
            approved = (APPROVED_TRAINING_COMMITS if approved_training_commits is None
                        else approved_training_commits)
            if sha not in approved:
                raise ValueError(
                    f"blob recipe git_sha {sha[:12]}… is not an APPROVED training "
                    "commit (populate APPROVED_TRAINING_COMMITS after the Tier-1 "
                    "launch commit is fixed)"
                )
            src_sha = str(recipe.get("source_checkpoint_sha256", ""))
            want_src = registered_arm1_sha256(source_seed)
            if src_sha != want_src:
                raise ValueError(
                    f"blob recipe source_checkpoint_sha256 {src_sha[:16]}… is not the "
                    f"registered arm1 checkpoint for seed s{source_seed}"
                )
            rc = str(recipe["source_checkpoint"])
            if rc != str(blob.get("source_checkpoint")):
                raise ValueError(
                    f"recipe source_checkpoint {rc!r} != blob source_checkpoint "
                    f"{blob.get('source_checkpoint')!r}"
                )
            if (f"phase1_seedconfirm_s{source_seed}/" not in rc
                    or "arm1/sigreg_l0.01" not in rc):
                raise ValueError(
                    f"recipe source_checkpoint {rc!r} does not name the arm1 "
                    f"sigreg_l0.01 checkpoint for seed s{source_seed}"
                )
            return
        if arm not in ("a1_adapted", "ce_continue"):
            raise ValueError(
                f"legacy adapted blob (no arm field) cannot be claimed as {arm!r}"
            )
        src_ckpt = str(blob.get("source_checkpoint", ""))
        if f"phase1_seedconfirm_s{source_seed}/" not in src_ckpt:
            raise ValueError(
                f"blob's embedded source_checkpoint {src_ckpt!r} does not name "
                f"claimed seed s{source_seed}"
            )
        if "arm1/sigreg_l0.01" not in src_ckpt:
            raise ValueError(
                f"blob's source_checkpoint {src_ckpt!r} is not the arm1 "
                "sigreg_l0.01 starting point"
            )
        allowlist = LEGACY_BLOB_SHA256 if legacy_allowlist is None else legacy_allowlist
        digest = model_sha256 or _sha256_file(path)
        entry = allowlist.get(digest)
        if entry is None:
            raise ValueError(
                f"legacy blob sha256 {digest[:16]}… is not one of the six "
                "frozen legacy artifacts"
            )
        if entry != (arm, source_seed):
            raise ValueError(
                f"legacy blob is registered as arm={entry[0]!r} seed {entry[1]}, "
                f"claimed arm={arm!r} seed {source_seed}"
            )
        if not LEGACY_GATE_SEMANTICS[arm](pr):
            raise ValueError(
                f"blob's embedded gate result (passed={pr.get('passed')}, "
                f"gate_value={pr.get('gate_value')}) is inconsistent with arm {arm!r}"
            )
        sidecar = path.parent / "config.json"
        if not sidecar.is_file():
            raise ValueError(f"legacy blob has no sidecar config.json at {sidecar}")
        side = json.loads(sidecar.read_text())
        lam = side.get("probe_cfg", {}).get("lambda_kv")
        if lam != LEGACY_SIDECAR_LAMBDA[arm]:
            raise ValueError(
                f"sidecar lambda_kv={lam} does not match arm {arm!r} "
                f"(expected {LEGACY_SIDECAR_LAMBDA[arm]})"
            )
        result_sidecar = path.parent / "probe_result.json"
        if not result_sidecar.is_file():
            raise ValueError("legacy blob has no sidecar probe_result.json")
        rs = json.loads(result_sidecar.read_text())
        if (rs.get("gate_value") != pr.get("gate_value")
                or rs.get("extra", {}).get("nll_after") != pr.get("extra", {}).get("nll_after")):
            raise ValueError(
                "blob's embedded probe_result does not match the sidecar "
                "probe_result.json — blob and directory are different runs"
            )
        if f"s{source_seed}" not in str(path):
            raise ValueError(f"path {path} does not name claimed seed s{source_seed}")
        return
    if continuation_seed is not None:
        raise ValueError(
            f"phase-1 checkpoints have no continuation seed; got {continuation_seed}"
        )
    if int(blob.get("global_step", -1)) != 610352:
        raise ValueError(
            f"trainer checkpoint global_step={blob.get('global_step')}, expected 610352"
        )
    cfg = blob.get("config")
    embedded_seed = getattr(cfg, "seed", None)
    if embedded_seed is None or int(embedded_seed) != int(source_seed):
        raise ValueError(
            f"trainer checkpoint embeds seed={embedded_seed}, claimed {source_seed}"
        )
    weights = {t.name: float(t.weight) for t in cfg.loss_terms}
    expected_weights = {
        "l_aux": {"l_aux": 1.0},
        "l_aux+l_sigreg": {"l_aux": 1.0, "l_sigreg": 0.01},
    }.get(arm)
    if expected_weights is None:
        raise ValueError(f"trainer checkpoint cannot be claimed as arm {arm!r}")
    if weights != expected_weights:
        raise ValueError(
            f"trainer checkpoint loss weights {weights} != registered arm "
            f"{arm!r} weights {expected_weights}"
        )
    tk = dict(getattr(cfg, "loss_term_kwargs", None) or {})
    if "l_sigreg" in weights:
        sig = dict(tk.get("l_sigreg") or {})
        if sig.get("key") != "h" or sig.get("m_projections") != 1024:
            raise ValueError(
                f"l_sigreg term kwargs {sig} != registered SIGReg(h) m=1024"
            )
    ce = dict(tk.get("l_aux") or {})
    if ce.get("label_smoothing", 0.0) != 0.0:
        raise ValueError(f"CE label_smoothing={ce.get('label_smoothing')} — expected 0.0")
    if ce.get("ignore_index", -100) != -100:
        raise ValueError(f"CE ignore_index={ce.get('ignore_index')} — expected -100")
    if (ce.get("logits_key", "logits"), ce.get("labels_key", "labels")) != ("logits", "labels"):
        raise ValueError("CE logits_key/labels_key differ from 'logits'/'labels'")
    registry = REGISTERED_MODEL_SHA256 if model_registry is None else model_registry
    digest = model_sha256 or _sha256_file(path)
    entry = registry.get(digest)
    if entry != (arm, source_seed):
        raise ValueError(
            f"trainer checkpoint sha256 {digest[:16]}… is registered as {entry}, "
            f"claimed ({arm!r}, {source_seed})"
        )


def _git_state() -> dict[str, Any]:
    import subprocess

    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=_REPO_ROOT, text=True).strip()
        except Exception as e:
            raise SystemExit(
                f"[sweep] git command {' '.join(cmd)} failed ({e}) — "
                "refusing to run with unknown code identity") from None

    sha = _run(["git", "rev-parse", "HEAD"])
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise SystemExit(f"[sweep] git sha {sha!r} is not 40-hex — aborting")
    return {"sha": sha, "dirty": bool(_run(["git", "status", "--porcelain"]))}


def _env_record(device: torch.device) -> dict[str, Any]:
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
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Registered group-size KV-quant sweep (one set).")
    p.add_argument("--model", required=True)
    p.add_argument("--arm", required=True, choices=KNOWN_ARMS)
    p.add_argument("--source-seed", type=int, required=True)
    p.add_argument("--continuation-seed", type=int, default=None,
                   help="42 for adapted/continuation blobs; omit for phase-1 checkpoints.")
    p.add_argument("--sweep", required=True, choices=sorted(SWEEPS))
    p.add_argument("--out-json", required=True)
    p.add_argument("--manifest", required=True,
                   help="preflight launch manifest authorizing EXACTLY this run")
    p.add_argument("--manifest-sha256", default=None,
                   help="expected manifest digest, passed through the preflight "
                        "exec argv — authenticates that THIS manifest is the one "
                        "preflight published")
    p.add_argument("--dev-dirty-ok", action="store_true",
                   help="dev only (propagated by preflight --allow-dirty): accept a "
                        "dirty manifest/checkout; result is tagged dev_mode and may "
                        "NOT target the registered output directory")
    p.add_argument("--tokenizer-name", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--eval-tokens", type=int, default=2_000_000)
    p.add_argument("--eval-docs", type=int, default=5000)
    p.add_argument("--eval-offset-docs", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--eval-seed", type=int, default=99)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    assert_pinned_protocol(args)
    out_path = Path(args.out_json)
    _is_canonical_pilot_run = (not args.dev_dirty_ok
                               and (args.arm, args.source_seed, args.sweep)
                               == (CANONICAL_PILOT_LAUNCH["arm"],
                                   CANONICAL_PILOT_LAUNCH["source_seed"],
                                   CANONICAL_PILOT_LAUNCH["sweep"]))
    assert_isolated_out_path(
        out_path, _REPO_ROOT, kind="sweep out-json",
        required_root=None if args.dev_dirty_ok else REGISTERED_SWEEP_OUT_ROOT,
        forbidden_root=REGISTERED_SWEEP_OUT_ROOT if args.dev_dirty_ok else None,
        allow_canonical_slot="pilot-result" if _is_canonical_pilot_run else None)

    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"[sweep] model not found: {model_path}")
    model_sha256 = _sha256_file(model_path)
    verify_model_provenance(
        model_path, arm=args.arm, source_seed=args.source_seed,
        continuation_seed=args.continuation_seed, model_sha256=model_sha256,
    )
    git = _git_state()
    manifest_path = Path(args.manifest)
    manifest_bytes = manifest_path.read_bytes()
    manifest = strict_json_loads(manifest_bytes.decode())
    verify_launch_manifest(
        manifest, manifest_path=args.manifest,
        model_path=str(model_path), model_sha256=model_sha256,
        arm=args.arm, source_seed=args.source_seed,
        continuation_seed=args.continuation_seed, sweep=args.sweep,
        out_json=str(out_path), head_sha=git["sha"], head_dirty=git["dirty"],
        dev_dirty_ok=args.dev_dirty_ok,
    )
    import hashlib as _hl
    manifest_sha256 = _hl.sha256(manifest_bytes).hexdigest()
    if args.manifest_sha256 is None:
        if not args.dev_dirty_ok:
            raise SystemExit(
                "[sweep] --manifest-sha256 is REQUIRED for retained runs (the "
                "preflight --launch wrapper supplies it); only explicit dev "
                "mode (--dev-dirty-ok) may omit it"
            )
    elif manifest_sha256 != args.manifest_sha256:
        raise SystemExit(
            f"[sweep] manifest sha256 {manifest_sha256[:16]}… != the digest "
            f"preflight passed through argv — not the published manifest"
        )
    if _is_canonical_pilot_run and manifest.get("mode") != "pilot-bootstrap":
        raise SystemExit(
            "[sweep] the canonical pilot tuple requires a verified "
            "pilot-bootstrap manifest — refusing to claim the canonical slot")

    assert_isolated_out_path(
        out_path, _REPO_ROOT, kind="sweep out-json",
        required_root=None if args.dev_dirty_ok else REGISTERED_SWEEP_OUT_ROOT,
        forbidden_root=REGISTERED_SWEEP_OUT_ROOT if args.dev_dirty_ok else None,
        allow_canonical_slot=("pilot-result"
                              if (not args.dev_dirty_ok
                                  and manifest.get("mode") == "pilot-bootstrap")
                              else None))
    lock, lock_token = claim_output_lock_isolated(_REPO_ROOT, out_path)
    try:
        if out_path.exists():
            raise SystemExit(f"[sweep] {out_path} exists — outputs are never "
                             "overwritten; delete or choose a fresh path")
        device = torch.device(args.device)
        enforce_single_visible_gpu(device)
        verified_pkgs = check_freeze_complete(
            _REPO_ROOT, rel_path=ANCHOR_FREEZE_REL_PATH,
            expected_sha256=ANCHOR_FREEZE_SHA256)
        if not args.dev_dirty_ok:
            assert_legacy_trees_locked(_REPO_ROOT)
        mpkgs = dict(manifest.get("packages") or {})
        if any(not isinstance(v, str) for v in mpkgs.values()) \
                or mpkgs != verified_pkgs:
            raise SystemExit(
                f"[sweep] manifest packages != verified freeze "
                f"({len(mpkgs)}/{len(verified_pkgs)} entries, literal strings "
                "required) — truncated or fabricated freeze evidence"
            )
        nvidia_smi_report = run_nvidia_smi()
        mcuda = manifest.get("cuda") or {}
        in_process_cuda = {
            "nvidia_smi": nvidia_smi_report,
            "device_name": (torch.cuda.get_device_name(0)
                            if device.type == "cuda" else "cpu"),
            "torch_cuda": str(getattr(torch.version, "cuda", None)),
            "visible_devices": (torch.cuda.device_count()
                                if device.type == "cuda" else 0),
        }
        cuda_wrong = {k: (mcuda.get(k), v) for k, v in in_process_cuda.items()
                      if mcuda.get(k) != v}
        if cuda_wrong:
            raise SystemExit(
                f"[sweep] manifest CUDA evidence disagrees with this host: "
                f"{cuda_wrong}"
            )

        tokenizer_revision = resolve_cached_revision("models", args.tokenizer_name)
        dataset_revision = resolve_cached_revision("datasets", args.dataset_name)
        tokenizer = load_pinned_tokenizer(
            args.tokenizer_name, tokenizer_revision, expected_vocab=32000)
        n_eval_batches = math.ceil(args.eval_tokens / (args.batch_size * args.seq_len))

        def raw_stream_factory():
            return load_pinned_fineweb(args.dataset_name, args.dataset_config,
                                       "train", dataset_revision)

        def tokenize_fn(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

        _train, eval_factory = build_fineweb_factories(
            raw_stream_factory, tokenize_fn,
            seq_len=args.seq_len, batch_size=args.batch_size,
            eos_token_id=tokenizer.eos_token_id,
            eval_docs=args.eval_docs, eval_offset_docs=args.eval_offset_docs,
            eval_n_batches=n_eval_batches,
        )

        trunk, head, loader_kind = kqe.load_model(model_path)
        kind = args.arm
        trunk.to(device).eval()
        head.to(device).eval()
        configs = SWEEPS[args.sweep]()
        print(f"[sweep] === {model_path} arm={args.arm} loader_kind={loader_kind} "
              f"sweep={args.sweep} ({len(configs)} configs, "
              f"{n_eval_batches} batches/eval) ===")
    except BaseException:
        release_output_lock(lock, lock_token)
        raise

    provenance = {
        "arm": args.arm,
        "source_seed": args.source_seed,
        "continuation_seed": args.continuation_seed,
        "eval_seed": args.eval_seed,
        "sweep": args.sweep,
    }
    rows: list[dict[str, Any]] = []
    reference_nll: float | None = None
    for cfg in configs:
        label = kqe.config_label(cfg)
        torch.manual_seed(args.eval_seed)
        counter = _CountingIter(eval_factory())
        t0 = time.time()
        if cfg["bits"] is None:
            metrics = evaluate(trunk, head, counter, device=device)
        else:
            with ExitStack() as stack:
                for ctx in kqe.quantizer_contexts(trunk, cfg):
                    stack.enter_context(ctx)
                metrics = evaluate(trunk, head, counter, device=device)
        if counter.count != n_eval_batches:
            raise AssertionError(
                f"[sweep] {label}: consumed {counter.count} batches, expected "
                f"{n_eval_batches} — dataset pass came up short; aborting unwritten"
            )
        nll = metrics["mean_nll"]
        if cfg["bits"] is None:
            reference_nll = nll
        rows.append({
            "model": str(model_path),
            "kind": kind,
            **provenance,
            "config": label,
            **cfg,
            "mean_nll": nll,
            "perplexity": metrics["perplexity"],
            "nll_delta_vs_ref": nll - reference_nll if reference_nll is not None else float("nan"),
            "anisotropy_h": metrics["anisotropy_h"],
            "n_batches": counter.count,
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"[sweep] {label:>34}: nll {nll:.4f} "
              f"(Δ {rows[-1]['nll_delta_vs_ref']:+.4f})  [{rows[-1]['elapsed_s']}s]")

    validate_sweep_rows(rows, sweep=args.sweep, expected_batches=n_eval_batches)
    git_now = _git_state()
    if git_now != git:
        raise SystemExit(
            f"[sweep] checkout changed during evaluation ({git} -> {git_now}) "
            "— refusing to publish"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert_lock_owned(lock, lock_token)
    env = _env_record(device)
    env["nvidia_smi"] = nvidia_smi_report
    env["tokenizer_repo"] = args.tokenizer_name
    env["tokenizer_revision"] = tokenizer_revision
    env["dataset_repo"] = args.dataset_name
    env["dataset_revision"] = dataset_revision
    env["model_sha256"] = model_sha256
    _is_canonical_pilot_run = (not args.dev_dirty_ok
                               and (args.arm, args.source_seed, args.sweep)
                               == (CANONICAL_PILOT_LAUNCH["arm"],
                                   CANONICAL_PILOT_LAUNCH["source_seed"],
                                   CANONICAL_PILOT_LAUNCH["sweep"]))
    assert_isolated_out_path(
        out_path, _REPO_ROOT, kind="sweep out-json",
        required_root=None if args.dev_dirty_ok else REGISTERED_SWEEP_OUT_ROOT,
        forbidden_root=REGISTERED_SWEEP_OUT_ROOT if args.dev_dirty_ok else None,
        allow_canonical_slot="pilot-result" if _is_canonical_pilot_run else None)
    publish_no_replace_isolated(_REPO_ROOT, out_path, json.dumps({
        "schema_version": "groupsweep-v1",
        "git": git,
        "command": sys.argv,
        "launch_manifest": str(manifest_path),
        "launch_manifest_sha256": manifest_sha256,
        "launch_manifest_json": manifest_bytes.decode(),
        "dev_mode": bool(args.dev_dirty_ok),
        "env": env,
        "rows": rows,
    }, indent=2, allow_nan=False))
    release_output_lock(lock, lock_token)
    print(f"[sweep] {len(rows)} validated rows -> {out_path}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
