from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import statistics
from pathlib import Path
from typing import Any

from jepa_lm.revision.envcheck import (
    ANCHOR_FREEZE_REL_PATH,
    ANCHOR_FREEZE_SHA256,
    load_verified_freeze,
)
from jepa_lm.revision.sweep_configs import (
    ANISO_REPLAY_ABS_TOL,
    CANONICAL_ANCHOR_CELLS,
    METRIC_TRIPLE,
    OVERLAP_CONFIGS,
    PINNED_N_BATCHES,
    REGISTERED_ARMS,
    REGISTERED_MODEL_SHA256,
    SWEEPS,
    label_for,
    merge_sweep_outputs,
    strict_json_loads,
    validate_merge_identity,
    validate_merged_envs,
    validate_split_payloads,
)

SCHEMA_VERSION = "groupsweep-merged-v1"
SPLITS = ("st3", "bt3", "bt4")

ARMS: tuple[str, ...] = (
    "l_aux",
    "l_aux+l_sigreg",
    "a1_adapted",
    "ce_continue",
    "ce_only",
    "h_k_only",
    "h_v_only",
)
assert set(ARMS) == set(REGISTERED_ARMS)
SEEDS: tuple[int, ...] = (42, 43, 44)

ARM_LABELS: dict[str, str] = {
    "l_aux": "Baseline",
    "l_aux+l_sigreg": "+SIGReg(h) pretrain",
    "a1_adapted": "A1 (h+K+V)",
    "ce_continue": "h-only cont.",
    "ce_only": "CE-only cont.",
    "h_k_only": "h+K cont.",
    "h_v_only": "h+V cont.",
}

LEGACY_ARMS: tuple[str, ...] = ("l_aux", "l_aux+l_sigreg", "a1_adapted",
                                "ce_continue")

PUBLISHED_BUNDLE_FILES: tuple[str, ...] = tuple(
    f"kv_quant/results_s{s}_{tag}.json"
    for s in SEEDS for tag in ("phase1", "a1", "ce_continue"))
EXPECTED_OVERLAP_MATCHED = 72
EXPECTED_OVERLAP_REFS = 12

OVERLAP_CELL_KEYS: tuple[CellKey, ...] = (
    (3, "per_channel", "both", "symmetric", None),
    (3, "per_token", "both", "symmetric", None),
    (4, "per_channel", "both", "symmetric", None),
    (4, "per_token", "both", "symmetric", None),
    (3, "per_token", "k", "symmetric", None),
    (3, "per_token", "v", "symmetric", None),
)

REGISTERED_MODEL_PATHS: dict[tuple[str, int], str] = {}
for _seed in (42, 43, 44):
    REGISTERED_MODEL_PATHS[("l_aux", _seed)] = (
        f"outputs/phase1_seedconfirm_s{_seed}/checkpoints/arm0/baseline/"
        "checkpoint_step610352.pt")
    REGISTERED_MODEL_PATHS[("l_aux+l_sigreg", _seed)] = (
        f"outputs/phase1_seedconfirm_s{_seed}/checkpoints/arm1/sigreg_l0.01/"
        "checkpoint_step610352.pt")
    REGISTERED_MODEL_PATHS[("a1_adapted", _seed)] = (
        f"outputs/phase1_5_replicates/s{_seed}_saved/a1/adapted_trunk.pt")
    REGISTERED_MODEL_PATHS[("ce_continue", _seed)] = (
        f"outputs/phase1_5_replicates/s{_seed}_ce_continue/a1/adapted_trunk.pt")
    for _arm in ("ce_only", "h_k_only", "h_v_only"):
        REGISTERED_MODEL_PATHS[(_arm, _seed)] = (
            f"outputs/revision/phase1_5_replicates/s{_seed}_{_arm}/a1/"
            "adapted_trunk.pt")

EXPECTED_CONTINUATION_SEED: dict[str, int | None] = {
    "l_aux": None, "l_aux+l_sigreg": None,
    "a1_adapted": 42, "ce_continue": 42,
    "ce_only": 42, "h_k_only": 42, "h_v_only": 42,
}


def _strict_int(v: Any) -> bool:
    return type(v) is int


LADDER_CELLS: tuple[tuple[Any, ...], ...] = (
    (3, "per_channel", "both", "symmetric", None),
    (3, "per_token", "both", "symmetric", None),
    (4, "per_channel", "both", "symmetric", None),
    (4, "per_token", "both", "symmetric", None),
    (3, "mixed", "both", "asymmetric", 32),
)

CellKey = tuple[Any, Any, Any, Any, Any]


class Pass1TableError(RuntimeError):
    pass


def cell_key(row: dict[str, Any]) -> CellKey:
    return (
        row.get("bits"),
        row.get("granularity"),
        row.get("targets"),
        row.get("mode"),
        row.get("group_size"),
    )


def _cfg_key(cfg: dict[str, Any]) -> CellKey:
    return (cfg.get("bits"), cfg.get("granularity"), cfg.get("targets"),
            cfg.get("mode"), cfg.get("group_size"))


def expected_cell_inventory() -> dict[CellKey, str]:
    inventory: dict[CellKey, str] = {}
    for sweep in SPLITS:
        for cfg in SWEEPS[sweep]():
            inventory[_cfg_key(cfg)] = label_for(cfg)
    if len(inventory) != 117:
        raise Pass1TableError(
            f"registered grid reconstruction yielded {len(inventory)} cells")
    return inventory


EXPECTED_CELLS = expected_cell_inventory()


def _expected_cell_sweeps() -> dict[CellKey, str]:
    out: dict[CellKey, str] = {}
    for sweep in SPLITS:
        for cfg in SWEEPS[sweep]():
            out.setdefault(_cfg_key(cfg), sweep)
    return out


EXPECTED_CELL_SWEEPS = _expected_cell_sweeps()

REQUIRED_ENV_KEYS = ("model_sha256", "torch", "dataset_repo",
                     "dataset_revision", "tokenizer_repo",
                     "tokenizer_revision")


def cell_label(key: CellKey) -> str:
    if key[0] is None:
        return "fp32-ref"
    return EXPECTED_CELLS[key]


def _require_hex64(value: Any, what: str, path: Path) -> None:
    if not (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)):
        raise Pass1TableError(f"{path}: {what} is not a 64-hex digest")


def load_merged(path: Path, *, arm: str, seed: int) -> dict[str, Any]:
    return validate_merged_bytes(path.read_bytes(), path, arm=arm, seed=seed)


def validate_merged_bytes(data: bytes, path: Path, *, arm: str,
                          seed: int) -> dict[str, Any]:
    payload = strict_json_loads(data.decode())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise Pass1TableError(f"{path}: schema_version != {SCHEMA_VERSION}")
    ref = payload.get("reference_nll")
    if not isinstance(ref, float) or not math.isfinite(ref):
        raise Pass1TableError(f"{path}: reference_nll not a finite float")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(SPLITS):
        raise Pass1TableError(f"{path}: sources must cover exactly {SPLITS}")
    for name in SPLITS:
        src = sources[name]
        _require_hex64(src.get("input_sha256"), f"sources.{name}.input_sha256",
                       path)
        _require_hex64(src.get("launch_manifest_sha256"),
                       f"sources.{name}.launch_manifest_sha256", path)
        git = src.get("git")
        if not isinstance(git, dict) or "sha" not in git \
                or git.get("dirty") is not False:
            raise Pass1TableError(
                f"{path}: sources.{name}.git evidence malformed "
                "(need dict with sha and literal dirty=False)")
        if src.get("dev_mode") is not False:
            raise Pass1TableError(
                f"{path}: sources.{name}.dev_mode must be literal False, "
                f"got {src.get('dev_mode')!r}")
        cmd = src.get("command")
        if not (isinstance(cmd, list) and cmd
                and all(isinstance(c, str) and c for c in cmd)):
            raise Pass1TableError(f"{path}: sources.{name}.command malformed")
    env = payload.get("env")
    if not isinstance(env, dict) or not env:
        raise Pass1TableError(f"{path}: merged env missing/empty")
    for key_ in REQUIRED_ENV_KEYS:
        if not (isinstance(env.get(key_), str) and env[key_]):
            raise Pass1TableError(f"{path}: env.{key_} missing/empty")
    if REGISTERED_MODEL_SHA256.get(env["model_sha256"]) != (arm, seed):
        raise Pass1TableError(
            f"{path}: env.model_sha256 does not map to ({arm}, s{seed}) in "
            "REGISTERED_MODEL_SHA256")

    rows = payload["rows"]
    seen: dict[CellKey, str] = {}
    fp32_rows = []
    for row in rows:
        key = cell_key(row)
        if key in seen:
            raise Pass1TableError(f"{path}: duplicate cell {key}")
        if key not in EXPECTED_CELLS and key != (None,) * 5:
            raise Pass1TableError(
                f"{path}: cell outside the registered grid: {key} "
                f"({row.get('config')!r})")
        seen[key] = row["config"]
        if not _strict_int(row["n_batches"]) \
                or row["n_batches"] != PINNED_N_BATCHES:
            raise Pass1TableError(
                f"{path}: n_batches {row['n_batches']!r} != {PINNED_N_BATCHES}"
                f" at {row['config']}")
        if row["arm"] != arm:
            raise Pass1TableError(f"{path}: arm {row['arm']!r} != {arm!r}")
        if not _strict_int(row["source_seed"]) or row["source_seed"] != seed:
            raise Pass1TableError(
                f"{path}: source_seed {row['source_seed']!r} != {seed}")
        if not _strict_int(row["eval_seed"]) or row["eval_seed"] != 99:
            raise Pass1TableError(f"{path}: eval_seed {row['eval_seed']!r}")
        expected_cont = EXPECTED_CONTINUATION_SEED[arm]
        if row["continuation_seed"] != expected_cont or (
                expected_cont is not None
                and not _strict_int(row["continuation_seed"])):
            raise Pass1TableError(
                f"{path}: continuation_seed {row['continuation_seed']!r} "
                f"!= {expected_cont!r} for arm {arm}")
        if row.get("bits") is not None and not _strict_int(row["bits"]):
            raise Pass1TableError(f"{path}: non-integer bits at "
                                  f"{row['config']}")
        if row.get("group_size") is not None \
                and not _strict_int(row["group_size"]):
            raise Pass1TableError(f"{path}: non-integer group_size at "
                                  f"{row['config']}")
        if row["model"] != REGISTERED_MODEL_PATHS[(arm, seed)]:
            raise Pass1TableError(
                f"{path}: model {row['model']!r} is not the registered path "
                f"for ({arm}, s{seed})")
        if row["kind"] != arm:
            raise Pass1TableError(
                f"{path}: kind {row['kind']!r} != arm {arm!r} at "
                f"{row['config']}")
        if row.get("sweep") != EXPECTED_CELL_SWEEPS[key]:
            raise Pass1TableError(
                f"{path}: sweep {row.get('sweep')!r} != registered "
                f"{EXPECTED_CELL_SWEEPS[key]!r} at {row['config']}")
        for metric in ("mean_nll", "nll_delta_vs_ref", "perplexity"):
            v = row[metric]
            if not isinstance(v, float) or not math.isfinite(v):
                raise Pass1TableError(
                    f"{path}: non-finite {metric} at {row['config']}")
        if row["perplexity"] != math.exp(row["mean_nll"]):
            raise Pass1TableError(
                f"{path}: perplexity != exp(mean_nll) at {row['config']}")
        if row["config"] == "fp32-ref":
            fp32_rows.append(row)
            continue
        if row["config"] != EXPECTED_CELLS[key]:
            raise Pass1TableError(
                f"{path}: config label {row['config']!r} != registered "
                f"{EXPECTED_CELLS[key]!r}")
        if row["nll_delta_vs_ref"] != row["mean_nll"] - ref:
            raise Pass1TableError(
                f"{path}: nll_delta_vs_ref inconsistent at {row['config']}")
    missing = set(EXPECTED_CELLS) - set(seen)
    if missing:
        raise Pass1TableError(f"{path}: registered cells missing: "
                              f"{sorted(missing, key=str)[:4]}...")
    if len(fp32_rows) != 1:
        raise Pass1TableError(f"{path}: {len(fp32_rows)} fp32-ref rows != 1")
    if fp32_rows[0]["mean_nll"] != ref:
        raise Pass1TableError(f"{path}: fp32 row != reference_nll")
    if fp32_rows[0]["nll_delta_vs_ref"] != 0.0:
        raise Pass1TableError(f"{path}: fp32 delta != 0")
    return payload


def authenticate_merged(groupsweep_dir: Path, *, arm: str, seed: int,
                        payload: dict[str, Any],
                        freeze_pins: dict[str, str]) -> None:
    raws: dict[str, bytes] = {}
    digests: dict[str, str] = {}
    for name in SPLITS:
        split_path = groupsweep_dir / f"results_s{seed}_{arm}__{name}.json"
        if not split_path.is_file():
            raise Pass1TableError(f"missing split for authentication: "
                                  f"{split_path}")
        data = split_path.read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        if digests[name] != payload["sources"][name]["input_sha256"]:
            raise Pass1TableError(
                f"{split_path}: bytes hash to {digests[name][:12]}…, not the "
                "merged artifact's recorded input_sha256")
        raws[name] = data
    split_payloads = {n: strict_json_loads(raws[n].decode()) for n in SPLITS}
    validate_split_payloads(split_payloads, freeze_pins=freeze_pins,
                            split_digests=digests)
    merged = merge_sweep_outputs(
        {n: p["rows"] for n, p in split_payloads.items()})
    merged_env = validate_merged_envs(
        {n: p.get("env") for n, p in split_payloads.items()})
    validate_merge_identity(merged_env, merged["rows"])
    if merged["rows"] != payload["rows"]:
        raise Pass1TableError(
            f"{arm}/s{seed}: merged rows do not re-derive from the "
            "authenticated splits")
    if merged["reference_nll"] != payload["reference_nll"]:
        raise Pass1TableError(
            f"{arm}/s{seed}: reference_nll does not re-derive")
    if merged_env != payload["env"]:
        raise Pass1TableError(f"{arm}/s{seed}: merged env does not re-derive")
    expected_sources = {
        name: {"git": p.get("git"), "command": p.get("command"),
               "dev_mode": p.get("dev_mode"),
               "input_sha256": digests[name],
               "launch_manifest": p.get("launch_manifest"),
               "launch_manifest_sha256": p.get("launch_manifest_sha256"),
               "launch_manifest_json": p.get("launch_manifest_json")}
        for name, p in split_payloads.items()}
    if payload["sources"] != expected_sources:
        raise Pass1TableError(
            f"{arm}/s{seed}: sources projection does not re-derive from the "
            "authenticated splits")
    row0 = merged["rows"][0]
    if row0["arm"] != arm or row0["source_seed"] != seed:
        raise Pass1TableError(
            f"{arm}/s{seed}: authenticated splits carry provenance "
            f"{row0['arm']}/s{row0['source_seed']}")


def load_all_merged(groupsweep_dir: Path,
                    repo_root: Path | None = None,
                    authenticate: bool = True
                    ) -> tuple[dict[tuple[str, int], dict[str, Any]],
                               dict[str, str]]:
    if repo_root is None:
        repo_root = groupsweep_dir.resolve().parents[3]
    freeze_pins = None
    if authenticate:
        freeze_pins = load_verified_freeze(
            repo_root, rel_path=ANCHOR_FREEZE_REL_PATH,
            expected_sha256=ANCHOR_FREEZE_SHA256)
    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    digests: dict[str, str] = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = groupsweep_dir / f"results_s{seed}_{arm}__merged.json"
            if not path.is_file():
                raise Pass1TableError(f"missing merged artifact: {path}")
            data = path.read_bytes()
            payload = validate_merged_bytes(data, path, arm=arm, seed=seed)
            if authenticate:
                authenticate_merged(groupsweep_dir, arm=arm, seed=seed,
                                    payload=payload,
                                    freeze_pins=freeze_pins)
            payloads[(arm, seed)] = payload
            digests[path.name] = hashlib.sha256(data).hexdigest()
    return payloads, digests


def aggregate(payloads: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    cells: dict[CellKey, dict[str, dict[str, Any]]] = {}
    refs: dict[str, dict[str, Any]] = {}
    for (arm, seed), payload in payloads.items():
        ref = payload["reference_nll"]
        refs.setdefault(arm, {"per_seed": {}})["per_seed"][seed] = ref
        for row in payload["rows"]:
            if row["config"] == "fp32-ref":
                continue
            entry = cells.setdefault(cell_key(row), {}).setdefault(
                arm, {"per_seed": {}})
            entry["per_seed"][seed] = {
                "delta": row["nll_delta_vs_ref"],
                "nll": row["mean_nll"],
            }
    for arm, summary in refs.items():
        vals = [summary["per_seed"][s] for s in SEEDS]
        summary["mean"] = statistics.fmean(vals)
        summary["stdev"] = statistics.stdev(vals)
    for entry_by_arm in cells.values():
        for arm, entry in entry_by_arm.items():
            deltas = [entry["per_seed"][s]["delta"] for s in SEEDS]
            nlls = [entry["per_seed"][s]["nll"] for s in SEEDS]
            entry["delta_mean"] = statistics.fmean(deltas)
            entry["delta_stdev"] = statistics.stdev(deltas)
            entry["nll_mean"] = statistics.fmean(nlls)
    return {"cells": cells, "refs": refs}


def _require_cell(agg: dict[str, Any], key: CellKey) -> dict[str, dict[str, Any]]:
    try:
        entry = agg["cells"][key]
    except KeyError:
        raise Pass1TableError(f"cell absent from grid: {key}")
    missing = [a for a in ARMS if a not in entry]
    if missing:
        raise Pass1TableError(f"cell {key} missing arms: {missing}")
    return entry


def scheme_preference_audit(agg: dict[str, Any]) -> dict[str, Any]:
    pc = _require_cell(agg, (3, "per_channel", "both", "symmetric", None))
    pt = _require_cell(agg, (3, "per_token", "both", "symmetric", None))
    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        rows = {}
        for seed in SEEDS:
            d_pc = pc[arm]["per_seed"][seed]["delta"]
            d_pt = pt[arm]["per_seed"][seed]["delta"]
            rows[seed] = {
                "delta_pc": d_pc,
                "delta_pt": d_pt,
                "prefers": "per_channel" if d_pc < d_pt else "per_token",
            }
        prefs = {rows[s]["prefers"] for s in SEEDS}
        per_arm[arm] = {
            "per_seed": rows,
            "consistent": len(prefs) == 1,
            "prefers": prefs.pop() if len(prefs) == 1 else "mixed",
        }
    pc_arms = [a for a in ARMS if per_arm[a]["prefers"] == "per_channel"]
    return {
        "per_arm": per_arm,
        "per_channel_preferring_arms": pc_arms,
        "a1_only_claim_holds": pc_arms == ["a1_adapted"],
    }


def best_vs_best_audit(agg: dict[str, Any], *, bits: int = 3,
                       targets: str = "both") -> dict[str, Any]:
    out: dict[str, Any] = {"bits": bits, "targets": targets,
                           "margin_convention":
                               "baseline_best_nll - arm_best_nll "
                               "(positive favors the arm)",
                           "per_arm": {}}
    for arm in ARMS:
        per_seed = {}
        for seed in SEEDS:
            best_key, best = None, None
            for key, entry_by_arm in agg["cells"].items():
                if key[0] != bits or arm not in entry_by_arm:
                    continue
                if targets is not None and key[2] != targets:
                    continue
                cand = entry_by_arm[arm]["per_seed"][seed]
                if best is None or cand["nll"] < best["nll"]:
                    best_key, best = key, cand
            per_seed[seed] = {
                "cell": cell_label(best_key),
                "delta": best["delta"],
                "nll": best["nll"],
            }
        out["per_arm"][arm] = per_seed
    base = out["per_arm"]["l_aux"]
    for arm in ARMS:
        for seed in SEEDS:
            entry = out["per_arm"][arm][seed]
            entry["margin_favor_arm"] = base[seed]["nll"] - entry["nll"]
    return out


def fmt_delta3(x: float) -> str:
    return f"{x:+.3f}"


def fmt_pm(mean: float, stdev: float) -> str:
    return f"{mean:+.3f}±{stdev:.3f}"


def fmt_pm_short(mean: float, stdev: float) -> str:
    sd = f"{stdev:.2f}".lstrip("0") or ".00"
    return f"{mean:+.2f}{{\\pm}}{sd}"


def fmt_ref3(x: float) -> str:
    return f"{x:.3f}"


def fmt_abs2(ref_mean: float, delta_mean: float) -> str:
    return f"{ref_mean + delta_mean:.2f}"


def fmt_abs3(ref_mean: float, delta_mean: float) -> str:
    return f"{ref_mean + delta_mean:.3f}"


def render_ladder_md(agg: dict[str, Any]) -> str:
    lines = [
        "| cell | " + " | ".join(ARM_LABELS[a] for a in ARMS) + " |",
        "|" + "---|" * (len(ARMS) + 1),
    ]
    ref = agg["refs"]
    lines.append(
        "| fp32 ref NLL | "
        + " | ".join(f"{ref[a]['mean']:.4f}±{ref[a]['stdev']:.4f}" for a in ARMS)
        + " |")
    for key in LADDER_CELLS:
        entry = _require_cell(agg, key)
        cells = [fmt_pm(entry[a]["delta_mean"], entry[a]["delta_stdev"])
                 for a in ARMS]
        lines.append(f"| {cell_label(key)} | " + " | ".join(cells) + " |")
        absr = [f"{ref[a]['mean'] + entry[a]['delta_mean']:.3f}" for a in ARMS]
        lines.append(f"| {cell_label(key)} abs NLL | " + " | ".join(absr) + " |")
    return "\n".join(lines)


def _group_sort(group: Any) -> tuple[int, int]:
    if group is None:
        return (1, 0)
    return (0, int(group))


def render_boundary_md(agg: dict[str, Any], *, bits: int, mode: str) -> str:
    keys = sorted(
        (k for k in agg["cells"]
         if k[0] == bits and k[2] == "both" and k[3] == mode),
        key=lambda k: (k[1], _group_sort(k[4])))
    lines = [
        "| granularity | group | "
        + " | ".join(ARM_LABELS[a] for a in ARMS) + " |",
        "|" + "---|" * (len(ARMS) + 2),
    ]
    for key in keys:
        entry = _require_cell(agg, key)
        cells = [fmt_pm(entry[a]["delta_mean"], entry[a]["delta_stdev"])
                 for a in ARMS]
        group = "free" if key[4] is None else key[4]
        lines.append(f"| {key[1]} | {group} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_single_target_md(agg: dict[str, Any]) -> str:
    keys = sorted(
        (k for k in agg["cells"] if k[2] in ("k", "v")),
        key=lambda k: (k[2], k[1], k[3], _group_sort(k[4])))
    lines = [
        "| target | granularity | mode | group | "
        + " | ".join(ARM_LABELS[a] for a in ARMS) + " |",
        "|" + "---|" * (len(ARMS) + 4),
    ]
    for key in keys:
        entry = _require_cell(agg, key)
        cells = [fmt_pm(entry[a]["delta_mean"], entry[a]["delta_stdev"])
                 for a in ARMS]
        group = "free" if key[4] is None else key[4]
        lines.append(
            f"| {key[2]} | {key[1]} | {key[3]} | {group} | "
            + " | ".join(cells) + " |")
    return "\n".join(lines)


TEX_CELL_LABELS: dict[CellKey, str] = {
    (3, "per_channel", "both", "symmetric", None):
        "3-bit per-channel (sym., group-free)",
    (3, "per_token", "both", "symmetric", None):
        "3-bit per-token (sym., group-free)",
    (4, "per_channel", "both", "symmetric", None):
        "4-bit per-channel (sym., group-free)",
    (4, "per_token", "both", "symmetric", None):
        "4-bit per-token (sym., group-free)",
    (3, "mixed", "both", "asymmetric", 32):
        "3-bit KIVI-style (mixed, asym., $g{=}32$)",
    (4, "mixed", "both", "asymmetric", 32):
        "4-bit KIVI-style (mixed, asym., $g{=}32$)",
}


def ladder_table_cells(agg: dict[str, Any],
                       july: dict[str, Any] | None = None
                       ) -> dict[str, list[str]]:
    ref = agg["refs"]
    out: dict[str, list[str]] = {
        "fp32": [fmt_ref3(ref[a]["mean"]) for a in ARMS]}
    if july is not None:
        bk = july["by_kind"]
        dash = ["---"] * (len(ARMS) - len(LEGACY_ARMS))
        out["8bit"] = [f"+{bk[a][8]['delta_mean']:.3f}".replace("+0.", "+.")
                       for a in LEGACY_ARMS] + dash
        out["8bit/abs"] = [f"{bk[a][8]['nll_mean']:.2f}"
                           for a in LEGACY_ARMS] + dash
        out["2bit"] = [fmt_pm_short(bk[a][2]["delta_mean"],
                                    bk[a][2]["delta_stdev"])
                       for a in LEGACY_ARMS] + dash
        out["2bit/abs"] = [f"{bk[a][2]['nll_mean']:.2f}"
                           for a in LEGACY_ARMS] + dash
    for key in LADDER_CELLS:
        entry = _require_cell(agg, key)
        label = EXPECTED_CELLS[key]
        if key[1] == "mixed":
            out[label] = [f"+{entry[a]['delta_mean']:.3f}".replace("+0.", "+.")
                          for a in ARMS]
        else:
            out[label] = [fmt_pm_short(entry[a]["delta_mean"],
                                       entry[a]["delta_stdev"]) for a in ARMS]
        out[label + "/abs"] = [
            fmt_abs2(ref[a]["mean"], entry[a]["delta_mean"]) for a in ARMS]
    return out


def render_ladder_tex(agg: dict[str, Any],
                      july: dict[str, Any] | None = None) -> str:
    cells = ladder_table_cells(agg, july)

    def _cell(v: str) -> str:
        return v if v == "---" else f"${v}$"

    def _pair(label: str, key: str) -> list[str]:
        return [f"{label} & " + " & ".join(_cell(v) for v in cells[key])
                + r" \\",
                r"\quad absolute NLL & "
                + " & ".join(_cell(v) for v in cells[key + "/abs"])
                + r" \\"]

    rows = ["fp32 reference NLL & "
            + " & ".join(f"${v}$" for v in cells["fp32"]) + r" \\"]
    if july is not None:
        rows += _pair("8-bit per-channel (sym., group-free)", "8bit")
    order = [(4, "per_channel", "both", "symmetric", None),
             (4, "per_token", "both", "symmetric", None),
             (3, "per_channel", "both", "symmetric", None),
             (3, "per_token", "both", "symmetric", None)]
    for key in order:
        rows += _pair(TEX_CELL_LABELS[key], EXPECTED_CELLS[key])
    if july is not None:
        rows += _pair("2-bit per-channel (sym., group-free)", "2bit")
    kivi = (3, "mixed", "both", "asymmetric", 32)
    rows += _pair(TEX_CELL_LABELS[kivi], EXPECTED_CELLS[kivi])
    return "\n".join(rows)


def load_bundle_manifest(repo_root: Path,
                         bundle_root: Path | None = None
                         ) -> tuple[dict[str, str], str]:
    if bundle_root is None:
        bundle_root = repo_root / "paper" / "artifacts"
    manifest_path = bundle_root / "manifest.json"
    data = manifest_path.read_bytes()
    manifest = strict_json_loads(data.decode())
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list):
        raise Pass1TableError("bundle manifest lacks a files list")
    for e in files:
        if not (isinstance(e, dict) and isinstance(e.get("path"), str)
                and e["path"] and isinstance(e.get("sha256"), str)):
            raise Pass1TableError(
                "bundle manifest entry malformed (need path + sha256 "
                "strings)")
        if e["path"].startswith("/") or ".." in e["path"].split("/"):
            raise Pass1TableError(
                f"bundle manifest path escapes the bundle root: "
                f"{e['path']!r}")
    paths = [e["path"] for e in files]
    if len(paths) != len(set(paths)):
        raise Pass1TableError("bundle manifest declares duplicate paths")
    return ({e["path"]: e["sha256"] for e in files},
            hashlib.sha256(data).hexdigest())


def verify_bundle_closure(bundle_root: Path,
                          manifest: dict[str, str]) -> None:
    root_st = os.lstat(bundle_root)
    if stat.S_ISLNK(root_st.st_mode) or not stat.S_ISDIR(root_st.st_mode):
        raise Pass1TableError(
            f"bundle root must be a real (non-symlink) directory: "
            f"{bundle_root}")
    actual: set[str] = set()
    dirs: list[str] = []
    for cur, subdirs, files in os.walk(bundle_root, followlinks=False):
        for name in subdirs + files:
            p = Path(cur) / name
            rel = str(p.relative_to(bundle_root))
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                raise Pass1TableError(f"bundle contains a symlink: {rel}")
            if stat.S_ISDIR(st.st_mode):
                dirs.append(rel)
            elif stat.S_ISREG(st.st_mode):
                if st.st_nlink != 1:
                    raise Pass1TableError(
                        f"bundle contains a hardlinked file "
                        f"(st_nlink={st.st_nlink}): {rel}")
                actual.add(rel)
            else:
                raise Pass1TableError(
                    f"bundle contains a special file (mode "
                    f"{stat.filemode(st.st_mode)}): {rel}")
    declared = set(manifest) | {"manifest.json"}
    missing = sorted(declared - actual)
    extras = sorted(actual - declared)
    if missing:
        raise Pass1TableError(
            f"bundle closure: {len(missing)} declared file(s) absent, "
            f"e.g. {missing[:3]}")
    if extras:
        raise Pass1TableError(
            f"bundle closure: {len(extras)} unmanifested file(s) present, "
            f"e.g. {extras[:3]}")
    for d in dirs:
        if not any(a.startswith(d + "/") for a in actual):
            raise Pass1TableError(
                f"bundle closure: directory carries no manifested files: {d}")
    for rel in sorted(manifest):
        got = hashlib.sha256((bundle_root / rel).read_bytes()).hexdigest()
        if got != manifest[rel]:
            raise Pass1TableError(
                f"bundle closure: {rel} bytes do not match the manifest "
                f"digest (got {got[:12]}..., manifest "
                f"{manifest[rel][:12]}...)")


def read_bundle_bytes(bundle_root: Path, rel: str,
                      manifest: dict[str, str]) -> bytes:
    if rel not in manifest:
        raise Pass1TableError(f"bundle manifest has no entry for {rel}")
    path = bundle_root / rel
    if not path.is_file():
        raise Pass1TableError(f"bundle evidence missing: {rel}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest[rel]:
        raise Pass1TableError(
            f"bundle file {rel} digest {digest[:12]}… != manifest")
    return data


_SIDECAR_ARM_REQUIRED = ("ce_only", "h_k_only", "h_v_only")


SIDECAR_TOP_KEYS = frozenset({
    "probe", "n_steps_trained", "gate_metric", "gate_value",
    "gate_threshold", "passed", "per_layer", "last_train_loss", "extra",
    "elapsed_seconds"})
SIDECAR_PER_LAYER_KEYS = frozenset(
    f"{t}_{m}_{w}" for t in ("k", "v") for m in ("aniso", "top1")
    for w in ("before", "after"))
SIDECAR_EXTRA_KEYS_REVISION = frozenset({
    "aniso_after", "aniso_before", "aniso_before_reliable", "aniso_floor",
    "arm", "base_objective", "dropped_terms", "effective_lambda_kv",
    "gate_is_informational", "nll_after", "nll_before", "nll_delta",
    "nll_delta_threshold", "spectrum_top1_after", "spectrum_top1_before"})
SIDECAR_EXTRA_KEYS_LEGACY = frozenset({
    "aniso_after", "aniso_before", "aniso_before_reliable", "aniso_floor",
    "anisotropy_ok", "nll_after", "nll_before", "nll_delta",
    "nll_delta_threshold", "nll_ok", "spectrum_top1_after",
    "spectrum_top1_before"})


def validate_sidecar(sidecar: dict[str, Any], arm: str) -> None:
    if not isinstance(sidecar, dict):
        raise Pass1TableError("sidecar is not a JSON object")
    if set(sidecar) != SIDECAR_TOP_KEYS:
        raise Pass1TableError(
            f"sidecar top-level keys must be exactly "
            f"{sorted(SIDECAR_TOP_KEYS)}, got {sorted(sidecar)}")
    for key in ("probe", "gate_metric"):
        if not isinstance(sidecar[key], str) or not sidecar[key]:
            raise Pass1TableError(f"sidecar {key} not a non-empty string")
    for key in ("gate_value", "gate_threshold", "last_train_loss",
                "elapsed_seconds"):
        if not isinstance(sidecar[key], float) \
                or not math.isfinite(sidecar[key]):
            raise Pass1TableError(f"sidecar {key} not a finite float")
    if sidecar["elapsed_seconds"] <= 0:
        raise Pass1TableError("sidecar elapsed_seconds not positive")
    if type(sidecar["passed"]) is not bool:
        raise Pass1TableError("sidecar passed not a literal bool")
    if sidecar["passed"] is not (
            sidecar["gate_value"] >= sidecar["gate_threshold"]):
        raise Pass1TableError(
            "sidecar passed inconsistent with gate_value vs gate_threshold")
    if not _strict_int(sidecar["n_steps_trained"]) \
            or sidecar["n_steps_trained"] <= 0:
        raise Pass1TableError(
            "sidecar n_steps_trained not a positive literal int")
    extra = sidecar["extra"]
    if not isinstance(extra, dict):
        raise Pass1TableError("sidecar extra is not a JSON object")
    expected_extra = (SIDECAR_EXTRA_KEYS_REVISION
                      if arm in _SIDECAR_ARM_REQUIRED
                      else SIDECAR_EXTRA_KEYS_LEGACY)
    if set(extra) != expected_extra:
        raise Pass1TableError(
            f"sidecar extra keys for {arm} must be exactly "
            f"{sorted(expected_extra)}, got {sorted(extra)}")
    if not isinstance(extra.get("nll_delta"), float) \
            or not math.isfinite(extra["nll_delta"]):
        raise Pass1TableError("sidecar nll_delta not a finite float")
    if arm in _SIDECAR_ARM_REQUIRED and extra.get("arm") != arm:
        raise Pass1TableError(
            f"sidecar for {arm} must carry extra.arm == {arm!r}, got "
            f"{extra.get('arm')!r}")
    if "arm" in extra and extra["arm"] != arm:
        raise Pass1TableError(
            f"sidecar arm {extra['arm']!r} != expected {arm!r}")
    per_layer = sidecar["per_layer"]
    if not isinstance(per_layer, dict) \
            or set(per_layer) != SIDECAR_PER_LAYER_KEYS:
        raise Pass1TableError(
            f"sidecar per_layer keys must be exactly "
            f"{sorted(SIDECAR_PER_LAYER_KEYS)}")
    for key, vals in per_layer.items():
        if not (isinstance(vals, list) and len(vals) == 12
                and all(isinstance(v, float) and math.isfinite(v)
                        for v in vals)):
            raise Pass1TableError(f"sidecar per_layer[{key!r}] malformed")


def load_all_merged_bundle(bundle_root: Path, manifest: dict[str, str]
                           ) -> tuple[dict[tuple[str, int], dict[str, Any]],
                                      dict[str, str]]:
    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    digests: dict[str, str] = {}
    for arm in ARMS:
        for seed in SEEDS:
            rel = f"kv_quant/groupsweep/results_s{seed}_{arm}__merged.json"
            data = read_bundle_bytes(bundle_root, rel, manifest)
            payloads[(arm, seed)] = validate_merged_bytes(
                data, bundle_root / rel, arm=arm, seed=seed)
            digests[rel] = hashlib.sha256(data).hexdigest()
    return payloads, digests


_RECEIPT_TOP_KEYS = frozenset({
    "schema_version", "timestamp_utc", "git", "pilot", "published",
    "kind", "arm", "source_seed", "cells", "verdict", "command"})
_RECEIPT_PUBLISHED_PATH = "paper/artifacts/kv_quant/results_s42_phase1.json"
CANONICAL_ANCHOR_SHA256 = \
    "b68409cac3d53199bfbb5345ec13b7ccc7bf270f97b985903228c23b6ef5973a"
_RECEIPT_PILOT_PATH_PREFIX = \
    "outputs/revision/kv_quant/groupsweep/results_s42_l_aux__st3"


def load_bundle_receipt(bundle_root: Path,
                        manifest: dict[str, str]) -> dict[str, Any]:
    rel = "kv_quant/groupsweep/pilot_crosscheck_receipt.json"
    data = read_bundle_bytes(bundle_root, rel, manifest)
    receipt = strict_json_loads(data.decode())
    if not isinstance(receipt, dict) \
            or set(receipt) != _RECEIPT_TOP_KEYS:
        raise Pass1TableError(
            f"{rel}: receipt top-level keys must be exactly "
            f"{sorted(_RECEIPT_TOP_KEYS)}")
    if receipt["schema_version"] != "crosscheck-receipt-v1":
        raise Pass1TableError(f"{rel}: unexpected schema_version "
                              f"{receipt['schema_version']!r}")
    if receipt["verdict"] != "PASS":
        raise Pass1TableError(f"{rel}: verdict {receipt['verdict']!r} "
                              "!= 'PASS'")
    if receipt["arm"] != "l_aux" or receipt["source_seed"] != 42 \
            or not _strict_int(receipt["source_seed"]) \
            or receipt["kind"] != "l_aux":
        raise Pass1TableError(f"{rel}: pilot identity is not (l_aux, s42)")
    if not (isinstance(receipt["timestamp_utc"], str)
            and receipt["timestamp_utc"]):
        raise Pass1TableError(f"{rel}: timestamp_utc missing/empty")
    git = receipt["git"]
    if not (isinstance(git, dict) and set(git) == {"sha", "dirty"}
            and git["dirty"] is False
            and isinstance(git["sha"], str) and git["sha"]):
        raise Pass1TableError(
            f"{rel}: git evidence malformed (need {{sha, dirty}} with "
            "literal dirty=False)")
    cmd = receipt["command"]
    if not (isinstance(cmd, list) and cmd
            and all(isinstance(c, str) and c for c in cmd)):
        raise Pass1TableError(f"{rel}: command malformed")
    published = receipt["published"]
    if not (isinstance(published, dict)
            and set(published) == {"path", "sha256"}
            and published["path"] == _RECEIPT_PUBLISHED_PATH):
        raise Pass1TableError(
            f"{rel}: published evidence must reference "
            f"{_RECEIPT_PUBLISHED_PATH}")
    if published["sha256"] != CANONICAL_ANCHOR_SHA256:
        raise Pass1TableError(
            f"{rel}: published.sha256 != the frozen canonical anchor "
            "digest")
    anchor_rel = "kv_quant/results_s42_phase1.json"
    if manifest.get(anchor_rel) != CANONICAL_ANCHOR_SHA256:
        raise Pass1TableError(
            f"{rel}: bundle manifest digest for {anchor_rel} != the "
            "frozen canonical anchor digest (the bundled anchor bytes "
            "are not the ones the receipt vouches for)")
    pilot = receipt["pilot"]
    if not (isinstance(pilot, dict)
            and set(pilot) == {"path", "sha256", "content"}):
        raise Pass1TableError(
            f"{rel}: pilot evidence must carry exactly "
            "{path, sha256, content}")
    if not (isinstance(pilot["path"], str)
            and pilot["path"].startswith(_RECEIPT_PILOT_PATH_PREFIX)):
        raise Pass1TableError(f"{rel}: pilot.path is not the canonical "
                              "s42 l_aux st3 sweep artifact")
    _require_hex64(pilot["sha256"], "receipt pilot.sha256",
                   bundle_root / rel)
    if not (isinstance(pilot["content"], str) and pilot["content"]):
        raise Pass1TableError(f"{rel}: pilot.content missing/empty")
    content = strict_json_loads(pilot["content"])
    if not isinstance(content, dict) \
            or content.get("schema_version") != "groupsweep-v1":
        raise Pass1TableError(
            f"{rel}: pilot.content is not a groupsweep-v1 payload")
    if content.get("dev_mode") is not False:
        raise Pass1TableError(
            f"{rel}: pilot.content dev_mode must be literal False")
    cgit = content.get("git")
    if not (isinstance(cgit, dict) and "sha" in cgit
            and cgit.get("dirty") is False):
        raise Pass1TableError(
            f"{rel}: pilot.content git evidence malformed")
    crows = content.get("rows")
    if not isinstance(crows, list) or not crows:
        raise Pass1TableError(f"{rel}: pilot.content rows missing")
    by_config: dict[str, list[dict[str, Any]]] = {}
    for row in crows:
        if not isinstance(row, dict):
            raise Pass1TableError(f"{rel}: pilot.content row not a dict")
        by_config.setdefault(row.get("config"), []).append(row)
    cells = receipt.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(OVERLAP_CONFIGS):
        raise Pass1TableError(
            f"{rel}: receipt cells must be exactly {sorted(OVERLAP_CONFIGS)}")
    for label, cell in cells.items():
        for side in ("published", "pilot"):
            triple = cell.get(side)
            if not isinstance(triple, dict) or set(triple) != set(METRIC_TRIPLE):
                raise Pass1TableError(
                    f"{rel}: cell {label} {side} must carry {METRIC_TRIPLE}")
            for metric, value in triple.items():
                if not isinstance(value, float) or not math.isfinite(value):
                    raise Pass1TableError(
                        f"{rel}: cell {label} {side}.{metric} not finite")
        anchor = CANONICAL_ANCHOR_CELLS[label]
        for metric in METRIC_TRIPLE:
            if cell["published"][metric] != anchor[metric]:
                raise Pass1TableError(
                    f"{rel}: cell {label} published {metric} != frozen "
                    "anchor constant")
        for metric in ("mean_nll", "perplexity"):
            if cell["pilot"][metric] != cell["published"][metric]:
                raise Pass1TableError(
                    f"{rel}: cell {label} pilot {metric} != published "
                    "(exact replay required)")
        if abs(cell["pilot"]["anisotropy_h"]
               - cell["published"]["anisotropy_h"]) > ANISO_REPLAY_ABS_TOL:
            raise Pass1TableError(
                f"{rel}: cell {label} anisotropy outside the frozen "
                "replay band")
        src_rows = by_config.get(label, [])
        if len(src_rows) != 1:
            raise Pass1TableError(
                f"{rel}: pilot.content must carry cell {label} exactly "
                f"once, found {len(src_rows)}")
        for metric in METRIC_TRIPLE:
            if src_rows[0].get(metric) != cell["pilot"][metric]:
                raise Pass1TableError(
                    f"{rel}: cell {label} pilot {metric} != the embedded "
                    "pilot.content value (receipt not bound to its own "
                    "evidence)")
    return receipt


def read_bundle_file(repo_root: Path, rel: str,
                     manifest: dict[str, str],
                     bundle_root: Path | None = None) -> Any:
    if bundle_root is None:
        bundle_root = repo_root / "paper" / "artifacts"
    return strict_json_loads(
        read_bundle_bytes(bundle_root, rel, manifest).decode())


def crosscheck_published(payloads: dict[tuple[str, int], dict[str, Any]],
                         repo_root: Path,
                         manifest: dict[str, str] | None = None,
                         bundle_root: Path | None = None
                         ) -> dict[str, Any]:
    new_by_model: dict[tuple[str, CellKey], float] = {}
    for payload in payloads.values():
        for row in payload["rows"]:
            if row["config"] == "fp32-ref":
                continue
            if row["mode"] != "symmetric" or row.get("group_size") is not None:
                continue
            new_by_model[(row["model"], cell_key(row))] = row["mean_nll"]
    fp32_by_model = {
        payload["rows"][0]["model"]: payload["reference_nll"]
        for payload in payloads.values()
    }
    legacy_models = {payloads[(arm, seed)]["rows"][0]["model"]
                     for arm in LEGACY_ARMS for seed in SEEDS}
    if len(legacy_models) != 12:
        raise Pass1TableError("crosscheck: expected 12 distinct legacy models")
    expected_refs = set(legacy_models)
    expected_matches = {(m, key) for m in legacy_models
                        for key in OVERLAP_CELL_KEYS}
    seen_refs: dict[str, int] = {}
    seen_matches: dict[tuple[str, CellKey], int] = {}
    mismatched: list[tuple[str, Any]] = []
    if manifest is None:
        manifest, _ = load_bundle_manifest(repo_root, bundle_root)
    for rel in PUBLISHED_BUNDLE_FILES:
        for row in read_bundle_file(repo_root, rel, manifest, bundle_root):
            model = row["model"]
            if row.get("config") == "fp32-ref":
                if model in expected_refs:
                    seen_refs[model] = seen_refs.get(model, 0) + 1
                    if row["mean_nll"] != fp32_by_model[model]:
                        mismatched.append((rel, "fp32-ref"))
                continue
            if row.get("bits") not in (3, 4):
                continue
            if not _strict_int(row["bits"]):
                raise Pass1TableError(
                    f"{rel}: July row bits {row['bits']!r} is not a "
                    "literal int")
            if "mode" in row or "group_size" in row:
                continue
            ident = (model, (row["bits"], row.get("granularity"),
                             row.get("targets"), "symmetric", None))
            if ident in expected_matches:
                seen_matches[ident] = seen_matches.get(ident, 0) + 1
                if row["mean_nll"] != new_by_model[ident]:
                    mismatched.append((rel, row.get("config")))
    if set(seen_matches) != expected_matches:
        missing = sorted(expected_matches - set(seen_matches), key=str)[:4]
        raise Pass1TableError(
            f"crosscheck matched {len(seen_matches)} of "
            f"{len(expected_matches)} required identities; missing e.g. "
            f"{missing}")
    if set(seen_refs) != expected_refs:
        raise Pass1TableError(
            f"crosscheck refs {len(seen_refs)} != {len(expected_refs)} "
            f"(missing {sorted(expected_refs - set(seen_refs))[:3]})")
    dupes = {k: v for k, v in {**{("ref", m): c for m, c in seen_refs.items()},
                               **seen_matches}.items() if v != 1}
    if dupes:
        raise Pass1TableError(
            f"crosscheck: published rows must appear exactly once per "
            f"identity; duplicated: {sorted(dupes, key=str)[:3]}")
    if mismatched:
        raise Pass1TableError(f"crosscheck mismatches: {mismatched}")
    if (len(seen_matches) != EXPECTED_OVERLAP_MATCHED
            or len(seen_refs) != EXPECTED_OVERLAP_REFS):
        raise Pass1TableError("crosscheck inventory sizes drifted from the "
                              "frozen 72+12 contract")
    return {"matched": len(seen_matches), "fp32_refs_checked": len(seen_refs),
            "mismatched": mismatched,
            "scope": ("mean_nll values only; digest-verified tracked bundle; "
                      "exact identity sets enforced")}


CONTROLS_TABLE_CONFIGS = (
    ("symmetric, per-token", "3bit-per_token-both"),
    ("symmetric, per-channel", "3bit-per_channel-both"),
    ("asymmetric, per-token", "3bit-per_token-both-asym"),
    ("asymmetric, per-channel", "3bit-per_channel-both-asym"),
    ("mean-sub $+$ sym., per-token", "3bit-per_token-both-meansub"),
    ("mean-sub $+$ sym., per-channel", "3bit-per_channel-both-meansub"),
)

REGISTERED_JULY_MODELS: dict[str, tuple[str, int]] = {}
for _seed in (42, 43, 44):
    REGISTERED_JULY_MODELS[
        f"outputs/phase1_seedconfirm_s{_seed}/checkpoints/arm0/baseline/"
        f"checkpoint_step610352.pt"] = ("l_aux", _seed)
    REGISTERED_JULY_MODELS[
        f"outputs/phase1_seedconfirm_s{_seed}/checkpoints/arm1/"
        f"sigreg_l0.01/checkpoint_step610352.pt"] = ("l_aux+l_sigreg",
                                                     _seed)
    REGISTERED_JULY_MODELS[
        f"outputs/phase1_5_replicates/s{_seed}_saved/a1/"
        f"adapted_trunk.pt"] = ("a1_adapted", _seed)
    REGISTERED_JULY_MODELS[
        f"outputs/phase1_5_replicates/s{_seed}_ce_continue/a1/"
        f"adapted_trunk.pt"] = ("ce_continue", _seed)

CONTROLS_BUNDLE_FILES = tuple(
    f"kv_quant/{name}" for seed in (42, 43, 44)
    for name in (f"results_s{seed}_phase1.json",
                 f"results_s{seed}_a1.json",
                 f"results_s{seed}_ce_continue.json",
                 f"controls_s{seed}.json",
                 f"controls_s{seed}_ce_continue.json"))


def july_controls_table_values(repo_root: Path,
                               manifest: dict[str, str] | None = None,
                               bundle_root: Path | None = None
                               ) -> dict[str, dict[str, float]]:
    if manifest is None:
        manifest, _ = load_bundle_manifest(repo_root, bundle_root)
    wanted = {cfg for _, cfg in CONTROLS_TABLE_CONFIGS}
    values: dict[tuple[str, str, str], float] = {}
    for rel in CONTROLS_BUNDLE_FILES:
        for row in read_bundle_file(repo_root, rel, manifest, bundle_root):
            if row.get("config") not in wanted \
                    or row.get("targets") != "both":
                continue
            if not _strict_int(row["bits"]) or row["bits"] != 3:
                raise Pass1TableError(
                    f"{rel}: controls row bits must be the literal int 3, "
                    f"got {row['bits']!r}")
            model = row["model"]
            ident = REGISTERED_JULY_MODELS.get(model)
            if ident is None:
                raise Pass1TableError(
                    f"{rel}: model not in the frozen July identity "
                    f"registry: {model!r}")
            kind, seed = ident
            key = (kind, str(seed), row["config"])
            if key in values:
                raise Pass1TableError(
                    f"{rel}: duplicate controls cell {key}")
            delta = row["nll_delta_vs_ref"]
            if not isinstance(delta, float) or not math.isfinite(delta):
                raise Pass1TableError(
                    f"{rel}: controls cell {key} delta not finite")
            values[key] = delta
    means: dict[str, dict[str, float]] = {}
    for _, cfg in CONTROLS_TABLE_CONFIGS:
        means[cfg] = {}
        for kind in LEGACY_ARMS:
            cell_vals = [values.get((kind, s, cfg))
                         for s in ("42", "43", "44")]
            if any(v is None for v in cell_vals):
                raise Pass1TableError(
                    f"controls grid: missing seeds for ({kind}, {cfg})")
            means[cfg][kind] = statistics.fmean(cell_vals)
    return means


def july_quant_table_values(repo_root: Path,
                            manifest: dict[str, str] | None = None,
                            bundle_root: Path | None = None
                            ) -> dict[str, Any]:
    if manifest is None:
        manifest, _ = load_bundle_manifest(repo_root, bundle_root)
    per_model: dict[str, dict[int, tuple[float, float]]] = {}
    refs: dict[str, float] = {}
    kind_by_model: dict[str, str] = {}
    for rel in PUBLISHED_BUNDLE_FILES:
        for row in read_bundle_file(repo_root, rel, manifest, bundle_root):
            model = row["model"]
            if "arm1/sigreg" in model:
                kind_by_model[model] = "l_aux+l_sigreg"
            elif "arm0/baseline" in model:
                kind_by_model[model] = "l_aux"
            elif "_ce_continue" in model:
                kind_by_model[model] = "ce_continue"
            else:
                kind_by_model[model] = "a1_adapted"
            if row.get("config") == "fp32-ref":
                refs[model] = row["mean_nll"]
                continue
            if row.get("bits") is not None and not _strict_int(row["bits"]):
                raise Pass1TableError(
                    f"{rel}: July row bits {row['bits']!r} is not a "
                    "literal int")
            if (row.get("granularity") == "per_channel"
                    and row.get("targets") == "both"
                    and "mode" not in row and "group_size" not in row):
                per_model.setdefault(model, {})[row["bits"]] = (
                    row["nll_delta_vs_ref"], row["mean_nll"])
    by_kind: dict[str, dict[int, dict[str, float]]] = {}
    for kind in LEGACY_ARMS:
        models = [m for m, k in kind_by_model.items() if k == kind]
        if len(models) != 3:
            raise Pass1TableError(f"july grid: {kind} has {len(models)} models")
        for bits in (8, 4, 3, 2):
            deltas = [per_model[m][bits][0] for m in models]
            nlls = [per_model[m][bits][1] for m in models]
            by_kind.setdefault(kind, {})[bits] = {
                "delta_mean": statistics.fmean(deltas),
                "delta_stdev": statistics.stdev(deltas),
                "nll_mean": statistics.fmean(nlls),
            }
    ratios: dict[int, tuple[float, float]] = {}
    base = [m for m, k in kind_by_model.items() if k == "l_aux"]
    a1 = [m for m, k in kind_by_model.items() if k == "a1_adapted"]

    def seed_of(m: str) -> str:
        match = re.search(r"s(4[234])", m)
        if match is None:
            raise Pass1TableError(f"cannot extract seed from model {m!r}")
        return match.group(1)

    base_by_seed = {seed_of(m): m for m in base}
    a1_by_seed = {seed_of(m): m for m in a1}
    if set(base_by_seed) != {"42", "43", "44"} or set(a1_by_seed) != {
            "42", "43", "44"}:
        raise Pass1TableError("july grid: seed coverage incomplete")
    for bits in (4, 3):
        rs = [per_model[base_by_seed[s]][bits][0]
              / per_model[a1_by_seed[s]][bits][0] for s in ("42", "43", "44")]
        ratios[bits] = (min(rs), max(rs))
    return {"by_kind": by_kind, "refs_by_model": refs,
            "ratio_range": {b: ratios[b] for b in (4, 3)}}
