from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REGISTERED_ARMS = (
    "l_aux", "l_aux+l_sigreg", "a1_adapted", "ce_continue",
    "ce_only", "h_k_only", "h_v_only",
)
_QUANTIZER_KEYS = ("bits", "granularity", "targets", "mode", "group_size",
                   "locus", "subtract_mean", "standardize", "rotate",
                   "layers")

REGISTERED_MODEL_SHA256 = {
    "b5beabb53cf34d030644a39d73030ea96a20e6f91ecc012c1e01ba8674fcc5c4": ("l_aux", 42),
    "8c40acf103d0499f3dbd9d2b1762c563ce192767f578e784f646d7379d2d6b42": ("l_aux", 43),
    "42ddf374318087b88affe40c4d1e6b20f05ecb824956aaeff429d185872c6e1b": ("l_aux", 44),
    "fea6477ceebfe6508df2fb7f0973acf1a747d08b2fc3a749f29a863b5c7a8363": ("l_aux+l_sigreg", 42),
    "718e24fc7d7e839d2d07990d051c097244b4071d8f922ba3372a0230ecf37e13": ("l_aux+l_sigreg", 43),
    "73eea46ec1b38f320a16c830fd3a54877d0e1d496adacde900b936d8eddf19a4": ("l_aux+l_sigreg", 44),
    "6a6e81e8e2687dda050f71f55083b2dcb744c7bfbdfb2ae4b7c4b32048c5de47": ("a1_adapted", 42),
    "7f3a673ed9798ad04957b6bcfde51fa89f7cab3d2310f6aa2a694afb8505ab56": ("a1_adapted", 43),
    "e78af4c762052f0d07a39524ba2981e207ac10dfa2ff7ae8d9b577677ae46280": ("a1_adapted", 44),
    "292d3c3c0063a7dd8101a919b6ad53b0005454e3d060a372b6084203a009daf8": ("ce_continue", 42),
    "5d0f53daa2d02ec1f8673234628285144386f7ed9555a5dbec0f21ad7c2e016a": ("ce_continue", 43),
    "064ea198a3e8c7237cd627f2403fff65117f53cce914c58e3a3dac587e8c7d6c": ("ce_continue", 44),
    "1a0e69a197071cd57ab0f104b8635a3401db2110e74ce36ec26c82ab357f9438": ("ce_only", 42),
    "63c53ab23472f1d9bc49c422840cd7aaa5626c5a031829086af90297c73d7087": ("h_k_only", 42),
    "39181f646d0d73f93a7a71ead1f63fc37b7a29b60f0c30291993b5acc0f5a860": ("h_v_only", 42),
    "65cc94616892bdda05ae9ea5eab03dcd109e1ae7b3778c33b93dbbc8b77c5022": ("ce_only", 43),
    "0ce1a1b39555d3a831ae2b5c220d510ac1fb6626ab6585bb87b6a2fc0a6556ee": ("h_k_only", 43),
    "a6c5ae0fc99e0a2439e515ef4a3bc7c36f64add4efc5d3e3b509a1c8cc9025e8": ("h_v_only", 43),
    "d3a2e77086c37fea8b31255d5547a88206ad4f2f0d71e30a891ba053ab6ceeec": ("ce_only", 44),
    "3454fb0d742c6365fc5533eb5f9fb311ed8e17083c09830ad6ca7ef9c72ff900": ("h_k_only", 44),
    "333091a27e27e1f634e7939b15c25e79eb94fa14604b93d327568f0bf3754038": ("h_v_only", 44),
}

GROUP_SIZES = (16, 32, 64, 128, 256)
SEQ_LEN_T = 2048

PINNED_PROTOCOL: dict[str, Any] = {
    "tokenizer_name": "mistralai/Mistral-7B-v0.1",
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_config": "sample-10BT",
    "eval_tokens": 2_000_000,
    "eval_docs": 5000,
    "eval_offset_docs": 100_000,
    "batch_size": 8,
    "seq_len": 2048,
    "eval_seed": 99,
    "device": "cuda",
}
PINNED_N_BATCHES = 123

ANISO_REPLAY_ABS_TOL = 5e-4

LAUNCH_MODES = ("receipt", "pilot-bootstrap", "dev")

REGISTERED_SWEEP_OUT_ROOT = "outputs/revision/kv_quant/groupsweep"
REGISTERED_CONTINUATION_OUT_ROOT = "outputs/revision/phase1_5_replicates"
REGISTERED_LAYERWISE_OUT_ROOT = "outputs/revision/kv_quant/layerwise"
REGISTERED_RECON_OUT_ROOT = "outputs/revision/kv_quant/recon"
REGISTERED_ATTENTION_OUT_ROOT = "outputs/revision/kv_quant/attention"
REGISTERED_GEOMETRY_OUT_ROOT = "outputs/revision/geometry_uniform"
LEGACY_OUTPUT_ROOTS = (
    "outputs/kv_quant",
    "outputs/phase1_5",
    "outputs/phase1_5_replicates",
    "outputs/phase1_seedconfirm_s42",
    "outputs/phase1_seedconfirm_s43",
    "outputs/phase1_seedconfirm_s44",
    "outputs/phase1_exitgate",
    "outputs/channel_stats",
    "outputs/zeroshot",
    "outputs/sts_probe",
    "outputs/hf_export",
    "outputs/env",
)


REVISION_OUT_ROOT = "outputs/revision"


def assert_isolated_out_path(
    path: Any, repo_root: Any, *, kind: str = "output",
    required_root: str | None = None, forbidden_root: str | None = None,
    allow_canonical_slot: str | None = None,
) -> None:
    from pathlib import Path as _Path

    raw = _Path(path)
    if any(part == ".." for part in raw.parts):
        raise SystemExit(
            f"[isolation] {kind} {path} contains '..' components — writers "
            "consume the raw path, so only normalized paths are accepted"
        )
    rp = raw.resolve()
    root = _Path(repo_root).resolve()
    for legacy in LEGACY_OUTPUT_ROOTS:
        lr = root / legacy
        if rp == lr or lr in rp.parents:
            raise SystemExit(
                f"[isolation] {kind} {path} is inside the FROZEN legacy paper "
                f"tree {legacy} — existing experiments are preserved for the "
                "paper; revision writes are confined to outputs/revision/**"
            )
    rev_root = root / REVISION_OUT_ROOT
    if rev_root.exists() and rev_root.resolve() != rev_root:
        raise SystemExit(
            f"[isolation] {REVISION_OUT_ROOT} contains a symlinked component "
            f"(resolves to {rev_root.resolve()}) — writes could be redirected; "
            "refusing"
        )
    inside_repo = rp == root or root in rp.parents
    inside_revision = rp == rev_root or rev_root in rp.parents
    if inside_repo and not inside_revision:
        raise SystemExit(
            f"[isolation] {kind} {path} is inside the repository but outside "
            f"{REVISION_OUT_ROOT}/** — revision writes never touch code, "
            "paper, docs, or legacy trees (disposable outputs may live "
            "outside the repo instead)"
        )
    if required_root is not None:
        rr = root / required_root
        if rr.exists() and rr.resolve() != rr:
            raise SystemExit(
                f"[isolation] registered root {required_root} contains a "
                f"symlinked component (resolves to {rr.resolve()}) — refusing"
            )
        if not (rp == rr or rr in rp.parents):
            raise SystemExit(
                f"[isolation] registered {kind} {path} must live inside "
                f"{required_root} — one registered tree, no strays"
            )
    if forbidden_root is not None:
        fr = root / forbidden_root
        if rp == fr or fr in rp.parents:
            raise SystemExit(
                f"[isolation] dev {kind} {path} may not occupy the registered "
                f"tree {forbidden_root} — disposable runs stay disposable"
            )
    slots = {
        "pilot-result": CANONICAL_PILOT_LAUNCH["out_json"],
        "pilot-manifest": CANONICAL_PILOT_MANIFEST_OUT,
        "receipt": CANONICAL_RECEIPT_PATH,
    }
    for slot_name, rel in slots.items():
        sp = (root / rel).resolve()
        if rp == sp:
            if allow_canonical_slot != slot_name:
                raise SystemExit(
                    f"[isolation] {kind} {path} targets the canonical "
                    f"single-shot {slot_name} slot ({rel}) — only its one "
                    "legitimate writer may create it; squatting would block "
                    "the real artifact"
                )
            continue
        lockp = sp.with_name(sp.name + ".lock")
        if (sp in rp.parents or rp == lockp or lockp in rp.parents):
            raise SystemExit(
                f"[isolation] {kind} {path} would occupy a blocking alias of "
                f"the canonical single-shot {slot_name} slot ({rel}) — writes "
                "below the slot filename or at/below its .lock path are "
                "refused for every caller"
            )
        if rp != rev_root and rp in sp.parents and rev_root in rp.parents:
            raise SystemExit(
                f"[isolation] {kind} {path} is an ANCESTOR path of the "
                f"canonical single-shot {slot_name} slot ({rel}) — writing a "
                "file there would occupy a directory name the slot needs"
            )


def assert_legacy_trees_locked(repo_root: Any) -> int:
    import os as _os
    from pathlib import Path as _Path

    root = _Path(repo_root).resolve()
    unlocked: list[str] = []
    checked = 0

    def _check(p: _Path) -> None:
        nonlocal checked
        checked += 1
        if _os.stat(p, follow_symlinks=False).st_mode & 0o222:
            unlocked.append(str(p.relative_to(root)))

    outputs = root / "outputs"
    if not outputs.is_dir():
        raise SystemExit("[isolation] outputs/ is missing — nothing to protect?")
    _check(outputs)
    symlinks: list[str] = []
    for legacy in LEGACY_OUTPUT_ROOTS:
        lp = root / legacy
        if not lp.exists():
            raise SystemExit(
                f"[isolation] frozen legacy tree {legacy} is MISSING — the "
                "paper record must be present and locked before registered runs"
            )
        if lp.is_symlink():
            symlinks.append(legacy)
            continue
        for dirpath, dirnames, filenames in _os.walk(lp):
            _check(_Path(dirpath))
            for name in (*dirnames, *filenames):
                p = _Path(dirpath) / name
                if p.is_symlink():
                    symlinks.append(str(p.relative_to(root)))
                elif name in filenames:
                    _check(p)
    if symlinks:
        raise SystemExit(
            f"[isolation] {len(symlinks)} symlink(s) inside frozen legacy "
            f"trees (first: {symlinks[:5]}) — a frozen paper record contains "
            "no symlinks; remove/replace them before registered runs"
        )
    agg = root / "outputs" / "phase1_seedconfirm_aggregate.json"
    if not agg.is_file():
        raise SystemExit(
            "[isolation] outputs/phase1_seedconfirm_aggregate.json is MISSING "
            "— the frozen paper record is incomplete; restore it before "
            "registered runs"
        )
    _check(agg)
    if unlocked:
        raise SystemExit(
            f"[isolation] {len(unlocked)} legacy entries are WRITABLE "
            f"(first: {unlocked[:5]}) — re-lock the frozen paper record "
            "(chmod -R a-w) before any registered launch"
        )
    return checked


CANONICAL_PILOT_LAUNCH: dict[str, Any] = {
    "model": ("outputs/phase1_seedconfirm_s42/checkpoints/arm0/baseline/"
              "checkpoint_step610352.pt"),
    "arm": "l_aux",
    "source_seed": 42,
    "continuation_seed": None,
    "sweep": "st3",
    "out_json": REGISTERED_SWEEP_OUT_ROOT + "/results_s42_l_aux__st3.json",
}
CANONICAL_PILOT_MANIFEST_OUT = "outputs/revision/logs/pilot_manifest.json"
CANONICAL_RECEIPT_PATH = "outputs/revision/logs/pilot_crosscheck_receipt.json"

RECEIPT_SCHEMA_VERSION = "crosscheck-receipt-v1"
CANONICAL_ANCHOR_PATH = "paper/artifacts/kv_quant/results_s42_phase1.json"
CANONICAL_ANCHOR_SHA256 = (
    "b68409cac3d53199bfbb5345ec13b7ccc7bf270f97b985903228c23b6ef5973a")
OVERLAP_CONFIGS = ("fp32-ref", "3bit-per_token-k", "3bit-per_token-v")
METRIC_TRIPLE = ("mean_nll", "perplexity", "anisotropy_h")
CANONICAL_ANCHOR_CELLS: dict[str, dict[str, float]] = {
    "fp32-ref": {"mean_nll": 3.0924153327941895,
                 "perplexity": 22.030224089857743,
                 "anisotropy_h": 0.0237380713224411},
    "3bit-per_token-k": {"mean_nll": 3.553407669067383,
                         "perplexity": 34.93215211228048,
                         "anisotropy_h": 0.037987370043992996},
    "3bit-per_token-v": {"mean_nll": 3.187260866165161,
                         "perplexity": 24.221989222818372,
                         "anisotropy_h": 0.03472758084535599},
}


def assert_pinned_protocol(args: Any) -> None:
    for key, want in PINNED_PROTOCOL.items():
        got = getattr(args, key)
        if got != want:
            raise SystemExit(
                f"[sweep] {key}={got!r} deviates from the PINNED protocol value "
                f"{want!r} — registered sweeps take no protocol overrides"
            )

EXPECTED_ROWS = {"st3": 41, "bt3": 39, "bt4": 39}

_TOKENIZER_REPO = "mistralai/Mistral-7B-v0.1"
_DATASET_REPO = "HuggingFaceFW/fineweb"


def _pinned_revisions() -> dict[str, str]:
    from jepa_lm.revision.hf_pins import PINNED_HF_REVISIONS

    return PINNED_HF_REVISIONS


APPROVED_TRAINING_COMMITS: frozenset[str] = frozenset({
    "be6f76cedafec136ac665590ce3087750c1c9e0b",
})


def frozen_new_arm_recipes() -> dict[str, dict[str, Any]]:
    from jepa_lm.revision.envcheck import TRAIN_FREEZE_SHA256

    pins = _pinned_revisions()
    return {
        arm: {
            "lr": 1e-4,
            "lambda_kv": 0.0 if arm == "ce_only" else 0.01,
            "m_projections": 256,
            "amp_dtype": "bf16",
            "unfreeze": "trunk",
            "batch_size": 8,
            "seq_len": 2048,
            "train_tokens": 500_000_000,
            "dataset_name": _DATASET_REPO,
            "dataset_config": "sample-10BT",
            "device_type": "cuda",
            "rng_policy": "checkpoint_restored",
            "tokenizer_name": _TOKENIZER_REPO,
            "vocab_size": 32000,
            "dataset_split": "train",
            "eval_tokens": 2_000_000,
            "eval_docs": 5000,
            "eval_offset_docs": 100_000,
            "aniso_seed": 1234,
            "git_dirty": False,
            "tokenizer_revision": pins[_TOKENIZER_REPO],
            "dataset_revision": pins[_DATASET_REPO],
            "dev_mode": False,
            "train_freeze_sha256": TRAIN_FREEZE_SHA256,
        }
        for arm in ("ce_only", "h_k_only", "h_v_only")
    }


RECIPE_PRESENCE_FIELDS = ("git_sha", "source_checkpoint", "source_checkpoint_sha256")


def registered_arm1_sha256(source_seed: int) -> str:
    for sha, (arm, seed) in REGISTERED_MODEL_SHA256.items():
        if arm == "l_aux+l_sigreg" and seed == source_seed:
            return sha
    raise KeyError(f"no registered arm1 checkpoint for seed {source_seed}")


def _cfg(bits: int, granularity: str, targets: str, mode: str,
         group_size: int | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "bits": bits, "granularity": granularity, "targets": targets, "mode": mode,
    }
    if group_size is not None:
        cfg["group_size"] = group_size
    return cfg


def groupsweep_st3() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [{"bits": None}]
    for targets in ("k", "v"):
        for mode in ("symmetric", "asymmetric"):
            for gran in ("per_token", "per_channel"):
                for gs in (None, 128, 64, 32, 16):
                    configs.append(_cfg(3, gran, targets, mode, gs))
    return configs


def _groupsweep_bt(bits: int) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [{"bits": None}]
    for mode in ("symmetric", "asymmetric"):
        for gs in (*GROUP_SIZES, None):
            configs.append(_cfg(bits, "per_token", "both", mode, gs))
        for gs in (*GROUP_SIZES, SEQ_LEN_T, None):
            configs.append(_cfg(bits, "per_channel", "both", mode, gs))
        for gs in (*GROUP_SIZES, None):
            configs.append(_cfg(bits, "mixed", "both", mode, gs))
    return configs


def groupsweep_bt3() -> list[dict[str, Any]]:
    return _groupsweep_bt(3)


def groupsweep_bt4() -> list[dict[str, Any]]:
    return _groupsweep_bt(4)


SWEEPS = {
    "st3": groupsweep_st3,
    "bt3": groupsweep_bt3,
    "bt4": groupsweep_bt4,
}


def label_for(cfg: dict[str, Any]) -> str:
    if cfg["bits"] is None:
        return "fp32-ref"
    label = f"{cfg['bits']}bit-{cfg['granularity']}-{cfg['targets']}"
    if cfg.get("mode", "symmetric") != "symmetric":
        label += "-asym"
    if cfg.get("group_size"):
        label += f"-g{cfg['group_size']}"
    return label


def expected_labels(sweep: str) -> list[str]:
    return [label_for(c) for c in SWEEPS[sweep]()]


def validate_sweep_rows(
    rows: list[dict[str, Any]],
    *,
    sweep: str,
    expected_batches: int,
) -> None:
    expected = EXPECTED_ROWS[sweep]
    if len(rows) != expected:
        raise AssertionError(f"{sweep}: {len(rows)} rows, expected {expected}")
    labels = [r["config"] for r in rows]
    dupes = {x for x in labels if labels.count(x) > 1}
    if dupes:
        raise AssertionError(f"{sweep}: duplicate config labels {sorted(dupes)}")
    if rows[0]["config"] != "fp32-ref":
        raise AssertionError(f"{sweep}: first row is {rows[0]['config']!r}, expected fp32-ref")
    bad = [r["config"] for r in rows
           if not _strict_int(r.get("n_batches"))
           or r.get("n_batches") != expected_batches]
    if bad:
        raise AssertionError(
            f"{sweep}: rows with n_batches != literal int {expected_batches}: {bad[:5]}"
        )
    import math as _math

    metric_keys = ("mean_nll", "perplexity", "nll_delta_vs_ref", "anisotropy_h")
    for r in rows:
        missing = [k for k in metric_keys if k not in r]
        if missing:
            raise AssertionError(
                f"{sweep}: row {r['config']!r} missing metrics {missing}"
            )
        bad = [
            k for k in metric_keys
            if isinstance(r[k], bool) or not isinstance(r[k], (int, float))
            or not _math.isfinite(r[k])
        ]
        if bad:
            raise AssertionError(
                f"{sweep}: row {r['config']!r} non-numeric/non-finite metrics {bad}"
            )
        if not -1.0 <= r["anisotropy_h"] <= 1.0:
            raise AssertionError(
                f"{sweep}: row {r['config']!r} anisotropy_h={r['anisotropy_h']} "
                "outside [-1, 1]"
            )
    reference = rows[0]["mean_nll"]
    for r in rows:
        if abs(r["nll_delta_vs_ref"] - (r["mean_nll"] - reference)) > 1e-9:
            raise AssertionError(
                f"{sweep}: row {r['config']!r} nll_delta_vs_ref="
                f"{r['nll_delta_vs_ref']} != mean_nll - reference "
                f"({r['mean_nll']} - {reference})"
            )
        if r["mean_nll"] < 0 or r["perplexity"] <= 0:
            raise AssertionError(
                f"{sweep}: row {r['config']!r} has impossible metrics "
                f"(mean_nll={r['mean_nll']}, perplexity={r['perplexity']})"
            )
        expected_ppl = _math.exp(r["mean_nll"])
        if abs(r["perplexity"] - expected_ppl) > 1e-12 * expected_ppl:
            raise AssertionError(
                f"{sweep}: row {r['config']!r} perplexity={r['perplexity']} "
                f"!= exp(mean_nll)={expected_ppl}"
            )


def validate_split_exact(name: str, rows: list[dict[str, Any]]) -> None:
    validate_sweep_rows(rows, sweep=name, expected_batches=PINNED_N_BATCHES)
    want = expected_labels(name)
    got = [r["config"] for r in rows]
    if got != want:
        diff = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
        raise AssertionError(
            f"{name}: row {diff} is {got[diff]!r}, registered cell is "
            f"{want[diff]!r} — split does not match SWEEPS[{name!r}]()"
        )
    for r, cfg in zip(rows, SWEEPS[name]()):
        if r.get("sweep") != name:
            raise AssertionError(
                f"{name}: row {r['config']!r} carries sweep={r.get('sweep')!r}"
            )
        for key in _QUANTIZER_KEYS:
            if key in cfg:
                if (key not in r or r[key] != cfg[key]
                        or type(r[key]) is not type(cfg[key])):
                    raise AssertionError(
                        f"{name}: row {r['config']!r} field {key}="
                        f"{r.get(key)!r} != registered {cfg[key]!r} "
                        "(exact value and type)"
                    )
            elif key in r:
                raise AssertionError(
                    f"{name}: row {r['config']!r} carries {key}={r[key]!r}; "
                    "the registered cell has no such key"
                )
    for r in rows:
        missing = [k for k in _PROV_KEYS if k not in r]
        if missing:
            raise AssertionError(f"{name}: row {r['config']!r} missing {missing}")
        if not _strict_int(r["source_seed"]) or not _strict_int(r["eval_seed"]):
            raise AssertionError(
                f"{name}: row {r['config']!r} source_seed/eval_seed must be "
                "literal integers (42.0 is not 42)"
            )
        if r["continuation_seed"] is not None and not _strict_int(r["continuation_seed"]):
            raise AssertionError(
                f"{name}: row {r['config']!r} continuation_seed must be a "
                "literal int or None"
            )
    idents = {tuple(r[k] for k in _PROV_KEYS) for r in rows}
    if len(idents) != 1:
        raise AssertionError(
            f"{name}: mixed provenance within the split ({len(idents)} distinct "
            "model/arm/seed tuples) — a split is ONE model/arm/seed evaluation"
        )
    _validate_provenance_tuple(next(iter(idents)))


def merge_sweep_outputs(
    rows_by_sweep: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if set(rows_by_sweep) != set(EXPECTED_ROWS):
        raise AssertionError(f"merge needs exactly {sorted(EXPECTED_ROWS)}, got "
                             f"{sorted(rows_by_sweep)}")
    for name, rows in rows_by_sweep.items():
        validate_split_exact(name, rows)
    idents = {
        tuple(r[k] for k in _PROV_KEYS)
        for rows in rows_by_sweep.values() for r in rows
    }
    if len(idents) != 1:
        raise AssertionError(
            f"mixed provenance across splits ({len(idents)} distinct "
            f"model/arm/seed tuples) — merge refuses cross-model input"
        )
    refs = {name: rows[0] for name, rows in rows_by_sweep.items()}
    pairs = {name: (ref["mean_nll"], ref["perplexity"]) for name, ref in refs.items()}
    if len(set(pairs.values())) != 1:
        raise AssertionError(
            f"fp32 NLL/perplexity disagree across splits: {pairs} — "
            "investigate nondeterminism before merging"
        )
    anisos = {name: ref["anisotropy_h"] for name, ref in refs.items()}
    if max(anisos.values()) - min(anisos.values()) > ANISO_REPLAY_ABS_TOL:
        raise AssertionError(
            f"fp32 anisotropy spread {anisos} exceeds the calibrated replay "
            f"band ({ANISO_REPLAY_ABS_TOL}) — beyond reservoir nondeterminism"
        )
    nlls = {name: ref["mean_nll"] for name, ref in refs.items()}
    merged = [refs["st3"]]
    for name in sorted(rows_by_sweep):
        merged.extend(rows_by_sweep[name][1:])
    labels = {r["config"] for r in merged}
    if len(labels) != 117 or len(merged) != 117:
        raise AssertionError(
            f"merged union has {len(merged)} rows / {len(labels)} distinct "
            "labels — the registered grid is exactly 117 distinct cells"
        )
    return {"reference_nll": nlls["st3"], "rows": merged}


MERGE_ENV_KEYS = (
    "python", "torch", "transformers", "datasets", "tokenizers", "torch_cuda",
    "device_name", "nvidia_smi", "visible_devices",
    "tokenizer_repo", "tokenizer_revision",
    "dataset_repo", "dataset_revision", "model_sha256",
)


def expected_sweep_argv_tail(
    *, model: str, arm: str, source_seed: int, continuation_seed: int | None,
    sweep: str, out_json: str, manifest: str, manifest_sha256: str,
) -> list[str]:
    tail = ["--model", model, "--arm", arm,
            "--source-seed", str(source_seed), "--sweep", sweep,
            "--out-json", out_json, "--manifest", manifest,
            "--manifest-sha256", manifest_sha256, "--device", "cuda"]
    if continuation_seed is not None:
        tail += ["--continuation-seed", str(continuation_seed)]
    return tail


def _argv_flag_value(cmd: list[Any], flag: str) -> str | None:
    hits: list[str] = []
    strs = [str(c) for c in cmd]
    for i, c in enumerate(strs):
        if c == flag:
            if i + 1 >= len(strs):
                return None
            hits.append(strs[i + 1])
        elif c.startswith(flag + "="):
            hits.append(c.split("=", 1)[1])
    return hits[0] if len(hits) == 1 else None


def _as_map(v: Any) -> Mapping[str, Any]:
    return v if isinstance(v, Mapping) else {}


def strict_json_loads(text: str) -> Any:
    import json as _json

    def _reject(const: str) -> Any:
        raise ValueError(f"non-standard JSON constant {const!r} refused")

    return _json.loads(text, parse_constant=_reject)


def validate_receipt_content(
    receipt: Mapping[str, Any], *, expect_git_sha: str,
) -> None:
    import hashlib as _hl
    import math as _math

    if not isinstance(receipt, Mapping):
        raise AssertionError("receipt is not a JSON object")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise AssertionError(
            f"receipt schema {receipt.get('schema_version')!r} != "
            f"{RECEIPT_SCHEMA_VERSION!r}")
    if receipt.get("verdict") != "PASS":
        raise AssertionError(f"receipt verdict {receipt.get('verdict')!r} != PASS")
    if not receipt.get("timestamp_utc") or not receipt.get("command"):
        raise AssertionError("receipt missing timestamp/command evidence")
    ident = (receipt.get("kind"), receipt.get("arm"), receipt.get("source_seed"))
    if ident != ("l_aux", "l_aux", 42) or not _strict_int(receipt.get("source_seed")):
        raise AssertionError(
            f"receipt identity {ident} is not the canonical pilot "
            "('l_aux', 'l_aux', 42) — receipts certify THE pilot, nothing else")
    pub = _as_map(receipt.get("published"))
    if (pub.get("path"), pub.get("sha256")) != (CANONICAL_ANCHOR_PATH,
                                                CANONICAL_ANCHOR_SHA256):
        raise AssertionError(
            f"receipt published anchor {pub!r} != the canonical tracked "
            f"artifact {CANONICAL_ANCHOR_PATH} @ {CANONICAL_ANCHOR_SHA256[:16]}…")
    cells = receipt.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != set(OVERLAP_CONFIGS):
        raise AssertionError(
            f"receipt cells {sorted(cells) if isinstance(cells, Mapping) else cells!r} "
            f"!= the overlap set {sorted(OVERLAP_CONFIGS)}")
    for cfg in OVERLAP_CONFIGS:
        entry = cells[cfg]
        if not isinstance(entry, Mapping) or set(entry) != {"published", "pilot"}:
            raise AssertionError(f"receipt cell {cfg!r} malformed: {entry!r}")
        if entry["published"] != CANONICAL_ANCHOR_CELLS[cfg]:
            raise AssertionError(
                f"receipt cell {cfg!r} published triple {entry['published']!r} "
                f"!= the frozen anchor values {CANONICAL_ANCHOR_CELLS[cfg]!r}")
        pil = entry["pilot"]
        if not isinstance(pil, Mapping) or set(pil) != set(METRIC_TRIPLE):
            raise AssertionError(f"receipt cell {cfg!r} pilot triple malformed: {pil!r}")
        want = CANONICAL_ANCHOR_CELLS[cfg]
        if (pil["mean_nll"] != want["mean_nll"]
                or pil["perplexity"] != want["perplexity"]):
            raise AssertionError(
                f"receipt cell {cfg!r} pilot NLL/ppl {pil!r} != the anchor "
                "values — NLL/perplexity replay bit-exactly, so a PASS receipt "
                "cannot record anything else")
        aniso = pil["anisotropy_h"]
        if (isinstance(aniso, bool) or not isinstance(aniso, (int, float))
                or not _math.isfinite(aniso)
                or abs(aniso - want["anisotropy_h"]) > ANISO_REPLAY_ABS_TOL):
            raise AssertionError(
                f"receipt cell {cfg!r} pilot anisotropy {aniso!r} is not a "
                f"finite number inside the calibrated replay band "
                f"({ANISO_REPLAY_ABS_TOL}) of {want['anisotropy_h']} — NaN "
                "compares False against any tolerance and is refused explicitly")
    git = _as_map(receipt.get("git"))
    if git.get("sha") != expect_git_sha or git.get("dirty") is not False:
        raise AssertionError(
            f"receipt git {git!r} != the authorizing manifest's clean commit "
            f"{expect_git_sha!r} — a receipt cannot vouch for another commit")
    pilot = _as_map(receipt.get("pilot"))
    if pilot.get("path") != CANONICAL_PILOT_LAUNCH["out_json"]:
        raise AssertionError(
            f"receipt pilot path {pilot.get('path')!r} != the canonical pilot "
            f"result {CANONICAL_PILOT_LAUNCH['out_json']!r}")
    pcontent = pilot.get("content")
    psha = str(pilot.get("sha256", ""))
    if not isinstance(pcontent, str) or not pcontent:
        raise AssertionError(
            "embedded pilot payload missing — a receipt must carry the exact "
            "bytes of the pilot result it certifies")
    if _hl.sha256(pcontent.encode()).hexdigest() != psha:
        raise AssertionError(
            "embedded pilot payload does not hash to the receipt's recorded "
            "pilot sha256 — invented pilot evidence")
    try:
        payload = strict_json_loads(pcontent)
    except ValueError:
        raise AssertionError(
            "embedded pilot payload is not strict JSON (NaN/Infinity refused)"
        ) from None
    if not isinstance(payload, Mapping):
        raise AssertionError("embedded pilot payload is not a JSON object")
    if payload.get("schema_version") != "groupsweep-v1":
        raise AssertionError(
            f"embedded pilot schema {payload.get('schema_version')!r} != groupsweep-v1")
    pgit = _as_map(payload.get("git"))
    if pgit.get("sha") != expect_git_sha or pgit.get("dirty") is not False:
        raise AssertionError(
            f"embedded pilot git {pgit!r} != the receipt's clean commit "
            f"{expect_git_sha!r} — an old-code pilot cannot ride a new receipt")
    if payload.get("dev_mode") is not False:
        raise AssertionError(
            f"embedded pilot dev_mode={payload.get('dev_mode')!r} — the "
            "canonical pilot is a registered run, explicitly False required")
    if payload.get("launch_manifest") != CANONICAL_PILOT_MANIFEST_OUT:
        raise AssertionError(
            f"embedded pilot launch_manifest {payload.get('launch_manifest')!r} "
            f"!= the canonical single-shot path {CANONICAL_PILOT_MANIFEST_OUT!r}")
    pmjson = payload.get("launch_manifest_json")
    if (not isinstance(pmjson, str) or not pmjson
            or _hl.sha256(pmjson.encode()).hexdigest()
            != str(payload.get("launch_manifest_sha256", ""))):
        raise AssertionError(
            "embedded pilot's manifest content missing or not hashing to its "
            "recorded launch_manifest_sha256")
    try:
        pmemb = strict_json_loads(pmjson)
    except ValueError:
        raise AssertionError("embedded pilot's manifest is not strict JSON") from None
    if not isinstance(pmemb, Mapping):
        raise AssertionError("embedded pilot's manifest is not a JSON object")
    if pmemb.get("mode") != "pilot-bootstrap":
        raise AssertionError(
            f"embedded pilot's manifest mode={pmemb.get('mode')!r} — receipts "
            "certify THE canonical BOOTSTRAP pilot, nothing else")
    try:
        validate_manifest_evidence(pmemb)
    except AssertionError as e:
        raise AssertionError(f"embedded pilot's manifest evidence: {e}") from None
    if _as_map(pmemb.get("git")).get("sha") != expect_git_sha:
        raise AssertionError(
            "embedded pilot's manifest commit != the receipt commit")
    penv = _as_map(payload.get("env"))
    from pathlib import PurePath as _PurePath

    pcmd = payload.get("command")
    expected_tail = expected_sweep_argv_tail(
        model=CANONICAL_PILOT_LAUNCH["model"], arm="l_aux", source_seed=42,
        continuation_seed=None, sweep="st3",
        out_json=CANONICAL_PILOT_LAUNCH["out_json"],
        manifest=CANONICAL_PILOT_MANIFEST_OUT,
        manifest_sha256=str(payload.get("launch_manifest_sha256", "")))
    if (not isinstance(pcmd, list) or len(pcmd) != 1 + len(expected_tail)
            or _PurePath(str(pcmd[0])).name != "kv_quant_sweep.py"
            or [str(c) for c in pcmd[1:]] != expected_tail):
        raise AssertionError(
            "embedded pilot's command is not the canonical registered sweep "
            f"invocation (expected kv_quant_sweep.py + {expected_tail!r}, "
            f"got {pcmd!r})")
    pmodels = {_as_map(m).get("path"): _as_map(m).get("sha256")
               for m in (pmemb.get("models") or [])
               if isinstance(m, Mapping)}
    if pmodels.get(CANONICAL_PILOT_LAUNCH["model"]) != penv.get("model_sha256"):
        raise AssertionError(
            "embedded pilot's manifest model record does not bind the "
            "result's model_sha256 at the canonical model path")
    pcuda = _as_map(pmemb.get("cuda"))
    for key in ("nvidia_smi", "device_name", "torch_cuda"):
        mv = pcuda.get(key)
        if not isinstance(mv, str) or mv != penv.get(key):
            raise AssertionError(
                f"embedded pilot cuda.{key}={mv!r} != its env "
                f"{key}={penv.get(key)!r} (literal string required)")
    vd = pcuda.get("visible_devices")
    if not _strict_int(vd) or vd != 1 or penv.get("visible_devices") != vd:
        raise AssertionError(
            f"embedded pilot visible_devices manifest={vd!r} / "
            f"env={penv.get('visible_devices')!r} — the pinned protocol is "
            "the literal int 1, agreeing across both records")
    ppkgs = _as_map(pmemb.get("packages"))
    core_wrong = {k: (ppkgs.get(k), penv.get(k))
                  for k in ("torch", "transformers", "datasets", "tokenizers")
                  if not isinstance(ppkgs.get(k), str)
                  or ppkgs.get(k) != penv.get(k)}
    if core_wrong:
        raise AssertionError(
            "embedded pilot core package claims disagree between manifest "
            f"and env: {core_wrong} (literal strings required)")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise AssertionError("embedded pilot payload carries no rows")
    try:
        validate_split_exact("st3", rows)
    except AssertionError as e:
        raise AssertionError(f"embedded pilot rows fail the exact split contract: {e}") from None
    r0 = rows[0]
    if (r0.get("arm"), r0.get("source_seed")) != ("l_aux", 42):
        raise AssertionError(
            f"embedded pilot rows claim ({r0.get('arm')!r}, "
            f"{r0.get('source_seed')!r}), not the canonical ('l_aux', 42)")
    if r0.get("model") != CANONICAL_PILOT_LAUNCH["model"]:
        raise AssertionError(
            f"embedded pilot rows carry model={r0.get('model')!r} != the "
            f"canonical pilot model {CANONICAL_PILOT_LAUNCH['model']!r}")
    msha_env = penv.get("model_sha256")
    if REGISTERED_MODEL_SHA256.get(msha_env) != ("l_aux", 42):
        raise AssertionError(
            f"embedded pilot model_sha256 {str(msha_env)[:16]}… is not the "
            "REGISTERED ('l_aux', 42) model")
    by_cfg = {r["config"]: r for r in rows}
    for cfg in OVERLAP_CONFIGS:
        file_triple = {k: by_cfg[cfg][k] for k in METRIC_TRIPLE}
        if cells[cfg]["pilot"] != file_triple:
            raise AssertionError(
                f"receipt cell {cfg!r} pilot triple {cells[cfg]['pilot']!r} != "
                f"the embedded pilot's own row {file_triple!r} — the cells "
                "must be a transcript of the embedded payload, not free text")


def validate_manifest_evidence(
    manifest: Mapping[str, Any], *, allow_dirty: bool = False,
    expected_packages: Mapping[str, str] | None = None,
) -> None:
    import re as _re
    from datetime import datetime
    from pathlib import PurePath

    from jepa_lm.revision.hf_pins import PINNED_HF_REVISIONS

    if not isinstance(manifest, Mapping):
        raise AssertionError("manifest is not a JSON object")
    if manifest.get("schema_version") != "launch-manifest-v1":
        raise AssertionError(
            f"manifest schema {manifest.get('schema_version')!r} unknown")
    mgit = _as_map(manifest.get("git"))
    msha_git = str(mgit.get("sha", ""))
    if len(msha_git) != 40 or any(c not in "0123456789abcdef" for c in msha_git):
        raise AssertionError(
            f"manifest git sha {msha_git!r} is not 40-hex — every other "
            "surface enforces well-formed commits; the manifest must too")
    if not allow_dirty and mgit.get("dirty") is not False:
        raise AssertionError("manifest records a dirty checkout")
    ts = str(manifest.get("timestamp_utc", ""))
    try:
        parsed = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise AssertionError(
            f"manifest timestamp {ts!r} is not a valid ISO-8601 UTC instant"
        ) from None
    if parsed.year < 2026:
        raise AssertionError(f"manifest timestamp {ts!r} predates the project")
    cmd = manifest.get("command")
    if (not isinstance(cmd, list) or not cmd
            or PurePath(str(cmd[0])).name != "preflight.py"
            and not any(PurePath(str(c)).name == "preflight.py" for c in cmd)):
        raise AssertionError(
            "manifest command is not a preflight.py invocation (basename match)")
    pkgs = manifest.get("packages")
    if not isinstance(pkgs, dict) or not pkgs:
        raise AssertionError("manifest carries no package-freeze evidence")
    for core in ("torch", "transformers", "datasets", "tokenizers"):
        if core not in pkgs:
            raise AssertionError(f"manifest package evidence missing {core!r}")
        if expected_packages is not None and core in expected_packages:
            if pkgs[core] != expected_packages[core]:
                raise AssertionError(
                    f"manifest package {core}={pkgs[core]!r} != expected "
                    f"{expected_packages[core]!r} — fabricated evidence"
                )
    smi = str(_as_map(manifest.get("cuda")).get("nvidia_smi", ""))
    if not _re.search(r".+,\s*\d+\.\d+", smi):
        raise AssertionError(
            f"manifest nvidia-smi evidence {smi!r} is not a 'name, driver' report")
    if manifest.get("hf_revisions") != PINNED_HF_REVISIONS:
        raise AssertionError(
            f"manifest hf_revisions {manifest.get('hf_revisions')} != pinned "
            f"{PINNED_HF_REVISIONS}")
    mode = manifest.get("mode")
    if mode not in LAUNCH_MODES:
        raise AssertionError(
            f"manifest mode={mode!r} is not a declared launch mode "
            f"{LAUNCH_MODES} — receiptless/undeclared manifests fail closed")
    if mode == "dev" and not allow_dirty:
        raise AssertionError(
            "manifest mode='dev' — dev launches are disposable and are never "
            "retained or merged")
    receipt_rec = manifest.get("crosscheck_receipt")
    if mode != "receipt":
        if receipt_rec is not None:
            raise AssertionError(
                f"mode={mode!r} manifests must not carry a crosscheck_receipt "
                "— a smuggled receipt cannot substitute for the declared mode")
    else:
        import hashlib as _hl

        if not isinstance(receipt_rec, Mapping):
            raise AssertionError(
                "mode='receipt' requires embedded crosscheck_receipt evidence "
                "(path + sha256 + full content)")
        if receipt_rec.get("path") != CANONICAL_RECEIPT_PATH:
            raise AssertionError(
                f"crosscheck_receipt path {receipt_rec.get('path')!r} != the "
                f"canonical receipt {CANONICAL_RECEIPT_PATH!r} — the wrapper "
                "path is pinned, not self-reported")
        content = receipt_rec.get("content")
        rsha = str(receipt_rec.get("sha256", ""))
        if not isinstance(content, str) or not content:
            raise AssertionError("crosscheck_receipt content missing/empty")
        if _hl.sha256(content.encode()).hexdigest() != rsha:
            raise AssertionError(
                "crosscheck_receipt content does not hash to its recorded "
                "sha256 — invented receipt evidence")
        try:
            parsed = strict_json_loads(content)
        except ValueError:
            raise AssertionError(
                "crosscheck_receipt content is not strict JSON "
                "(NaN/Infinity refused)") from None
        try:
            validate_receipt_content(
                parsed,
                expect_git_sha=str(_as_map(manifest.get("git")).get("sha")))
        except AssertionError as e:
            raise AssertionError(f"embedded crosscheck receipt invalid: {e}") from None
    if mode == "pilot-bootstrap":
        mlaunch = _as_map(manifest.get("launch"))
        if dict(mlaunch) != CANONICAL_PILOT_LAUNCH:
            raise AssertionError(
                "mode='pilot-bootstrap' authorizes ONLY the canonical pilot "
                f"launch {CANONICAL_PILOT_LAUNCH}, got {manifest.get('launch')!r}")
        if not _strict_int(mlaunch.get("source_seed")):
            raise AssertionError(
                f"pilot-bootstrap launch source_seed="
                f"{mlaunch.get('source_seed')!r} must be the literal int 42 "
                "(42.0 is not 42)")
        mo = _argv_flag_value(list(manifest.get("command") or []), "--manifest-out")
        if mo != CANONICAL_PILOT_MANIFEST_OUT:
            raise AssertionError(
                f"pilot-bootstrap --manifest-out {mo!r} != the canonical "
                f"single-shot path {CANONICAL_PILOT_MANIFEST_OUT!r}")


def validate_merged_envs(env_by_sweep: dict[str, Any]) -> dict[str, Any]:
    if set(env_by_sweep) != set(EXPECTED_ROWS):
        raise AssertionError(f"envs for exactly {sorted(EXPECTED_ROWS)} required, "
                             f"got {sorted(env_by_sweep)}")
    for name, env in env_by_sweep.items():
        if not isinstance(env, dict):
            raise AssertionError(f"{name}: missing environment record")
        empty = [k for k in MERGE_ENV_KEYS if not env.get(k)]
        if empty:
            raise AssertionError(f"{name}: environment missing/empty fields {empty}")
        vd = env.get("visible_devices")
        if not _strict_int(vd) or vd != 1:
            raise AssertionError(
                f"{name}: visible_devices={vd!r} — the registered protocol "
                "requires the literal integer 1 (single visible GPU)"
            )
    views = {name: tuple(env[k] for k in MERGE_ENV_KEYS)
             for name, env in env_by_sweep.items()}
    if len(set(views.values())) != 1:
        detail = {k: {n: env_by_sweep[n].get(k) for n in env_by_sweep}
                  for k in MERGE_ENV_KEYS
                  if len({str(env_by_sweep[n].get(k)) for n in env_by_sweep}) > 1}
        raise AssertionError(f"environments differ across splits: {detail}")
    from jepa_lm.revision.hf_pins import PINNED_HF_REVISIONS

    common = {k: env_by_sweep["st3"][k] for k in MERGE_ENV_KEYS}
    if common["tokenizer_repo"] != _TOKENIZER_REPO:
        raise AssertionError(f"tokenizer_repo={common['tokenizer_repo']!r} != {_TOKENIZER_REPO!r}")
    if common["dataset_repo"] != _DATASET_REPO:
        raise AssertionError(f"dataset_repo={common['dataset_repo']!r} != {_DATASET_REPO!r}")
    for repo_key, rev_key in (("tokenizer_repo", "tokenizer_revision"),
                              ("dataset_repo", "dataset_revision")):
        pinned = PINNED_HF_REVISIONS[common[repo_key]]
        if common[rev_key] != pinned:
            raise AssertionError(
                f"{rev_key}={common[rev_key]} != registered pin {pinned} "
                f"for {common[repo_key]}"
            )
    model_sha = common["model_sha256"]
    if model_sha not in REGISTERED_MODEL_SHA256:
        raise AssertionError(
            f"model_sha256 {str(model_sha)[:16]}… is not a registered pass-1 model"
        )
    return common


def validate_merge_identity(common_env: Mapping[str, Any], rows: list[dict[str, Any]]) -> None:
    entry = REGISTERED_MODEL_SHA256.get(common_env["model_sha256"])
    claim = (rows[0]["arm"], rows[0]["source_seed"])
    if entry != claim:
        raise AssertionError(
            f"model hash is registered as {entry}, rows claim {claim}"
        )


def _strict_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


_PROV_KEYS = ("model", "kind", "arm", "source_seed", "continuation_seed", "eval_seed")


def _validate_provenance_tuple(ident: tuple[Any, ...]) -> None:
    (model, kind, arm, source_seed, continuation_seed, eval_seed) = ident
    if not isinstance(model, str) or not model:
        raise AssertionError(f"model={model!r} — nonempty model identity required")
    if not _strict_int(source_seed) or not _strict_int(eval_seed):
        raise AssertionError(
            f"source_seed={source_seed!r}/eval_seed={eval_seed!r} must be "
            "literal integers (42.0 is not 42)"
        )
    if continuation_seed is not None and not _strict_int(continuation_seed):
        raise AssertionError(
            f"continuation_seed={continuation_seed!r} must be a literal int or None"
        )
    if arm not in REGISTERED_ARMS:
        raise AssertionError(f"arm={arm!r} is not a registered arm")
    if kind != arm:
        raise AssertionError(f"kind={kind!r} != arm={arm!r} — contradictory rows")
    if eval_seed != PINNED_PROTOCOL["eval_seed"]:
        raise AssertionError(
            f"eval_seed={eval_seed!r} != pinned {PINNED_PROTOCOL['eval_seed']}"
        )
    if source_seed not in (42, 43, 44):
        raise AssertionError(f"source_seed={source_seed!r} is not a registered seed")
    trainer_arm = arm in ("l_aux", "l_aux+l_sigreg")
    if trainer_arm and continuation_seed is not None:
        raise AssertionError(
            f"trainer arm {arm!r} cannot carry continuation_seed={continuation_seed!r}"
        )
    if not trainer_arm and continuation_seed != 42:
        raise AssertionError(
            f"adapted arm {arm!r} must carry continuation_seed=42, "
            f"got {continuation_seed!r}"
        )


def validate_split_payloads(
    payloads: dict[str, Any], *, freeze_pins: Mapping[str, str],
    split_digests: Mapping[str, str] | None = None,
) -> None:
    shas = set()
    receipt_shas: set[str] = set()
    receipt_pilot_shas: set[str] = set()
    bootstrap_names: set[str] = set()
    for name, payload in payloads.items():
        if payload.get("schema_version") != "groupsweep-v1":
            raise AssertionError(
                f"{name}: schema_version={payload.get('schema_version')!r} "
                "!= groupsweep-v1"
            )
        git = payload.get("git") or {}
        sha = str(git.get("sha", ""))
        if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            raise AssertionError(f"{name}: git sha {sha!r} is not 40-hex")
        if git.get("dirty") is not False:
            raise AssertionError(f"{name}: retained results require git dirty=False")
        shas.add(sha)
        mpath = payload.get("launch_manifest")
        msha = str(payload.get("launch_manifest_sha256", ""))
        if not mpath or len(msha) != 64 or any(c not in "0123456789abcdef" for c in msha):
            raise AssertionError(
                f"{name}: launch-manifest authorization evidence missing/invalid"
            )
        cmd = payload.get("command")
        if (not isinstance(cmd, list) or not cmd
                or not all(isinstance(c, str) and c for c in cmd)):
            raise AssertionError(f"{name}: launch command missing or implausible")
        mjson = payload.get("launch_manifest_json")
        if not isinstance(mjson, str) or not mjson:
            raise AssertionError(f"{name}: embedded launch-manifest content missing")
        import hashlib as _hl
        if _hl.sha256(mjson.encode()).hexdigest() != msha:
            raise AssertionError(
                f"{name}: embedded manifest content does not hash to "
                "launch_manifest_sha256 — invented evidence"
            )
        if payload.get("dev_mode") is not False:
            raise AssertionError(
                f"{name}: dev_mode={payload.get('dev_mode')!r} — must be "
                "explicitly False (dev results are never merged; missing/None/0 "
                "fails closed)"
            )
        try:
            memb = strict_json_loads(mjson)
        except ValueError:
            raise AssertionError(
                f"{name}: embedded manifest is not strict JSON "
                "(NaN/Infinity refused)") from None
        try:
            validate_manifest_evidence(memb, expected_packages=freeze_pins)
        except AssertionError as e:
            raise AssertionError(f"{name}: embedded manifest evidence: {e}") from None
        mpkgs = dict(memb.get("packages") or {})
        nonstr = {k: v for k, v in mpkgs.items() if not isinstance(v, str)}
        if nonstr:
            raise AssertionError(
                f"{name}: manifest package versions must be literal strings; "
                f"got {dict(list(nonstr.items())[:4])}"
            )
        if mpkgs != dict(freeze_pins):
            missing = sorted(set(freeze_pins) - set(mpkgs))[:4]
            extra = sorted(set(mpkgs) - set(freeze_pins))[:4]
            wrong = {k: (mpkgs.get(k), freeze_pins[k]) for k in freeze_pins
                     if k in mpkgs and mpkgs[k] != freeze_pins[k]}
            raise AssertionError(
                f"{name}: manifest packages != verified freeze "
                f"(missing {missing}, extra {extra}, "
                f"wrong {dict(list(wrong.items())[:4])}, "
                f"{len(mpkgs)}/{len(freeze_pins)} entries)"
            )
        penv = payload.get("env") or {}
        env_wrong = {k: (penv.get(k), freeze_pins.get(k))
                     for k in ("torch", "transformers", "datasets", "tokenizers")
                     if penv.get(k) != freeze_pins.get(k)}
        if env_wrong:
            raise AssertionError(
                f"{name}: env package versions disagree with the verified "
                f"freeze: {env_wrong}"
            )
        mcuda = memb.get("cuda") or {}
        for key in ("nvidia_smi", "device_name", "torch_cuda"):
            mv, ev = mcuda.get(key), penv.get(key)
            if not isinstance(mv, str) or mv != ev:
                raise AssertionError(
                    f"{name}: manifest cuda.{key}={mv!r} != result env "
                    f"{key}={ev!r} (literal string required)"
                )
        if mcuda.get("visible_devices") != penv.get("visible_devices") \
                or not _strict_int(mcuda.get("visible_devices")):
            raise AssertionError(
                f"{name}: manifest cuda.visible_devices="
                f"{mcuda.get('visible_devices')!r} != result env "
                f"{penv.get('visible_devices')!r} (literal int required)"
            )
        mgit = memb.get("git") or {}
        rows0 = payload["rows"][0]
        mlaunch = memb.get("launch") or {}
        binds = {
            "model": (mlaunch.get("model"), rows0.get("model")),
            "arm": (mlaunch.get("arm"), rows0.get("arm")),
            "source_seed": (mlaunch.get("source_seed"), rows0.get("source_seed")),
            "continuation_seed": (mlaunch.get("continuation_seed"),
                                  rows0.get("continuation_seed")),
            "sweep": (mlaunch.get("sweep"), rows0.get("sweep")),
        }
        mismatched = {k: v for k, v in binds.items() if v[0] != v[1]}
        if mismatched:
            raise AssertionError(
                f"{name}: embedded manifest binds a DIFFERENT run: {mismatched}"
            )
        if mgit.get("sha") != git.get("sha"):
            raise AssertionError(
                f"{name}: embedded manifest commit != payload commit"
            )
        menv = payload.get("env") or {}
        mmodels = {m.get("path"): m.get("sha256") for m in memb.get("models", [])}
        if mmodels.get(rows0.get("model")) != menv.get("model_sha256"):
            raise AssertionError(
                f"{name}: embedded manifest model digest != result model_sha256"
            )
        expected_tail = expected_sweep_argv_tail(
            model=str(rows0.get("model")), arm=str(rows0.get("arm")),
            source_seed=rows0.get("source_seed"),
            continuation_seed=rows0.get("continuation_seed"),
            sweep=str(rows0.get("sweep")),
            out_json=str(mlaunch.get("out_json")), manifest=str(mpath),
            manifest_sha256=msha,
        )
        from pathlib import PurePath
        if (len(cmd) != 1 + len(expected_tail)
                or PurePath(str(cmd[0])).name != "kv_quant_sweep.py"
                or list(cmd[1:]) != expected_tail):
            raise AssertionError(
                f"{name}: command is not the canonical registered invocation "
                f"(expected kv_quant_sweep.py + {expected_tail!r}, got {cmd!r}) "
                "— =-syntax variants, extras, and reordering all fail exact "
                "comparison"
            )
        env_sha = _as_map(payload.get("env")).get("model_sha256")
        is_canonical_identity = (
            str(rows0.get("sweep")) == CANONICAL_PILOT_LAUNCH["sweep"]
            and (REGISTERED_MODEL_SHA256.get(env_sha) == (
                    CANONICAL_PILOT_LAUNCH["arm"],
                    CANONICAL_PILOT_LAUNCH["source_seed"])
                 or (rows0.get("arm"), rows0.get("source_seed")) == (
                    CANONICAL_PILOT_LAUNCH["arm"],
                    CANONICAL_PILOT_LAUNCH["source_seed"])))
        if is_canonical_identity and memb.get("mode") != "pilot-bootstrap":
            raise AssertionError(
                f"{name}: rows/model registration identify THE canonical "
                "pilot (l_aux/42/st3), but the split is not its "
                "pilot-bootstrap artifact — receipt-mode relabeling of the "
                "pilot is refused (the byte binding must always apply)"
            )
        if memb.get("mode") == "pilot-bootstrap":
            if str(mpath) != CANONICAL_PILOT_MANIFEST_OUT:
                raise AssertionError(
                    f"{name}: bootstrap split consumed manifest {mpath!r}, not the "
                    f"canonical single-shot path {CANONICAL_PILOT_MANIFEST_OUT!r}")
            bootstrap_names.add(name)
        if memb.get("mode") == "receipt":
            rc = _as_map(memb.get("crosscheck_receipt"))
            receipt_shas.add(str(rc.get("sha256")))
            parsed_rc = strict_json_loads(str(rc.get("content")))
            receipt_pilot_shas.add(
                str(_as_map(_as_map(parsed_rc).get("pilot")).get("sha256")))
    if len(shas) != 1:
        raise AssertionError(f"splits were produced at different commits: {sorted(shas)}")
    if len(receipt_shas) > 1:
        raise AssertionError(
            f"splits carry {len(receipt_shas)} different crosscheck receipts "
            "— every receipt-mode split must embed the one canonical receipt")
    if bootstrap_names and receipt_pilot_shas:
        want = next(iter(receipt_pilot_shas))
        if split_digests is None:
            raise AssertionError(
                "byte digests of the split files are required to bind the "
                "bootstrap split to the receipt-certified pilot — pass "
                "split_digests (the merge CLI reads each split once as bytes)"
            )
        for name in sorted(bootstrap_names):
            if split_digests.get(name) != want:
                raise AssertionError(
                    f"{name}: bootstrap split bytes hash to "
                    f"{str(split_digests.get(name))[:16]}…, but the shared "
                    f"receipt certifies pilot {want[:16]}… — the merged pilot "
                    "must BE the receipt-certified pilot, byte for byte"
                )
