from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from jepa_lm.revision.pass1_tables import (
    ARM_LABELS,
    ARMS,
    LADDER_CELLS,
    aggregate,
    best_vs_best_audit,
    cell_label,
    crosscheck_published,
    july_quant_table_values,
    load_all_merged,
    load_all_merged_bundle,
    load_bundle_manifest,
    render_boundary_md,
    render_ladder_md,
    render_ladder_tex,
    render_single_target_md,
    scheme_preference_audit,
    verify_bundle_closure,
)
from jepa_lm.revision.guards import open_isolated_root
from jepa_lm.revision.sweep_configs import (
    REVISION_OUT_ROOT,
    assert_isolated_out_path,
)


def _atomic_publish_isolated(repo_root: Path, rel_path: str,
                             text: str) -> None:
    rel = Path(rel_path)
    dirfd = open_isolated_root(repo_root, rel.parent)
    try:
        tmp_name = rel.name + f".tmp.{os.getpid()}"
        fd = os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                     0o644, dir_fd=dirfd)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            os.unlink(tmp_name, dir_fd=dirfd)
            raise
        os.replace(tmp_name, rel.name, src_dir_fd=dirfd, dst_dir_fd=dirfd)
    finally:
        os.close(dirfd)


def _git(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=_REPO_ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _git_optional(args: list[str]) -> str | None:
    try:
        return _git(args)
    except (OSError, subprocess.CalledProcessError):
        return None


REGISTERED_ANALYSIS_DIR = "outputs/revision/analysis"
BUNDLE_ANALYSIS_DIR = "outputs/revision/analysis_bundle"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groupsweep-dir",
                   default="outputs/revision/kv_quant/groupsweep")
    p.add_argument("--evidence", choices=("full", "bundle"), default="full",
                   help="full (default): the raw merged artifacts, "
                        "authenticated against the complete output tree. "
                        "bundle: the curated copies in the artifact bundle, "
                        "digest-verified against its manifest (no raw-chain "
                        "replay).")
    p.add_argument("--bundle-root", default=None,
                   help="bundle mode: root of the artifact bundle "
                        "(default paper/artifacts)")
    p.add_argument("--out-dir", default=None,
                   help=f"default {REGISTERED_ANALYSIS_DIR} (full) or "
                        f"{BUNDLE_ANALYSIS_DIR} (bundle)")
    args = p.parse_args(argv)
    bundle_mode = args.evidence == "bundle"
    if args.bundle_root is not None and not bundle_mode:
        p.error("--bundle-root is only meaningful with --evidence bundle")

    gs_dir = Path(args.groupsweep_dir)
    if not gs_dir.is_absolute():
        gs_dir = _REPO_ROOT / gs_dir
    out_dir = Path(args.out_dir if args.out_dir is not None else
                   (BUNDLE_ANALYSIS_DIR if bundle_mode
                    else REGISTERED_ANALYSIS_DIR))
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    if bundle_mode and (out_dir.resolve()
                        == (_REPO_ROOT / REGISTERED_ANALYSIS_DIR).resolve()):
        raise SystemExit(
            f"refusing --out-dir {REGISTERED_ANALYSIS_DIR} with --evidence "
            "bundle: that directory holds the registered full-evidence "
            f"analysis; use {BUNDLE_ANALYSIS_DIR} (the default)")
    out_files = {
        "md": out_dir / "pass1_tables.md",
        "json": out_dir / "pass1_tables.json",
        "ladder_tex": out_dir / "tex" / "ladder_body.tex",
    }
    for f in out_files.values():
        assert_isolated_out_path(f, _REPO_ROOT, kind="analysis out",
                                 required_root=REVISION_OUT_ROOT)

    bundle_root: Path | None = None
    if bundle_mode:
        bundle_root = Path(args.bundle_root if args.bundle_root is not None
                           else "paper/artifacts")
        if not bundle_root.is_absolute():
            bundle_root = _REPO_ROOT / bundle_root
        manifest_map, manifest_digest = load_bundle_manifest(_REPO_ROOT,
                                                             bundle_root)
        verify_bundle_closure(bundle_root, manifest_map)
        payloads, merged_digests = load_all_merged_bundle(bundle_root,
                                                          manifest_map)
    else:
        payloads, merged_digests = load_all_merged(gs_dir)
    agg = aggregate(payloads)
    pref = scheme_preference_audit(agg)
    best3 = best_vs_best_audit(agg, bits=3, targets="both")
    best4 = best_vs_best_audit(agg, bits=4, targets="both")
    best3_full = best_vs_best_audit(agg, bits=3, targets=None)
    if not bundle_mode:
        manifest_map, manifest_digest = load_bundle_manifest(_REPO_ROOT)
    xck = crosscheck_published(payloads, _REPO_ROOT, manifest=manifest_map,
                               bundle_root=bundle_root)
    july = july_quant_table_values(_REPO_ROOT, manifest=manifest_map,
                                   bundle_root=bundle_root)

    if bundle_mode:
        git_sha = _git_optional(["rev-parse", "HEAD"])
        status = (_git_optional(["status", "--porcelain"])
                  if git_sha is not None else None)
        git_dirty = None if status is None else bool(status)
    else:
        git_sha = _git(["rev-parse", "HEAD"])
        git_dirty = bool(_git(["status", "--porcelain"]))
    provenance = {
        "generator": "scripts/revision/pass1_tables.py",
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "inputs_sha256": {
            **merged_digests,
            "paper/artifacts/manifest.json": manifest_digest,
        },
    }
    if bundle_mode:
        provenance["evidence"] = "bundle"
        provenance["raw_chain_replayed"] = False

    md = ["# Pass-1 analysis tables (21 models × 117 cells, n=3 per arm)",
          "",
          "ΔNLL in nats vs each model's own fp32 reference on the pinned",
          "2M-token window; mean±sample-std over source seeds 42/43/44.",
          "", "## Control ladder (headline cells)", "",
          render_ladder_md(agg), "",
          "## Locality boundary — 3-bit symmetric, both targets", "",
          render_boundary_md(agg, bits=3, mode="symmetric"), "",
          "## Locality boundary — 3-bit asymmetric, both targets", "",
          render_boundary_md(agg, bits=3, mode="asymmetric"), "",
          "## Locality boundary — 4-bit symmetric, both targets", "",
          render_boundary_md(agg, bits=4, mode="symmetric"), "",
          "## Locality boundary — 4-bit asymmetric, both targets", "",
          render_boundary_md(agg, bits=4, mode="asymmetric"), "",
          "## Single-target 3-bit grid (K-only and V-only diagnostics)", "",
          render_single_target_md(agg), "",
          "## Result-contingent claim audits", "",
          f"- per-channel-preferring arms (3-bit sym group-free): "
          f"{pref['per_channel_preferring_arms']}",
          f"- \"A1 is the only per-channel-preferring condition\" holds: "
          f"**{pref['a1_only_claim_holds']}**", "",
          "### Best-vs-best, deployment menu (both-target cells only)", "",
          "Margins use the paper's convention: baseline's best NLL minus",
          "the arm's best NLL — POSITIVE FAVORS THE ARM. Point estimates;",
          "the paired bootstrap over this enlarged pool needs per-batch",
          "records the sweep does not emit.", ""]
    for arm in ARMS:
        rows = best3["per_arm"][arm]
        md.append(f"- best 3-bit both-target cell ({ARM_LABELS[arm]}): "
                  + "; ".join(
            f"s{s}: {rows[s]['cell']} Δ{rows[s]['delta']:+.3f} "
            f"(abs {rows[s]['nll']:.3f}, margin "
            f"{rows[s]['margin_favor_arm']:+.3f})"
            for s in (42, 43, 44)))
    md += ["", "## Crosscheck vs published July artifacts (tracked bundle)",
           "",
           f"- {xck['matched']} overlapping mean-NLL values and "
           f"{xck['fp32_refs_checked']} fp32 reference NLLs matched "
           "bit-exactly (digest-verified bundle; scope: mean-NLL values — "
           "the pilot receipt separately certifies NLL/perplexity exactly "
           "and anisotropy within the frozen replay band).", ""]
    _atomic_publish_isolated(_REPO_ROOT,
                             out_files["md"].relative_to(_REPO_ROOT),
                             "\n".join(md))

    _atomic_publish_isolated(_REPO_ROOT,
                             out_files["ladder_tex"].relative_to(_REPO_ROOT),
                             render_ladder_tex(agg, july))

    payload = {
        "schema_version": "pass1-tables-v2",
        "provenance": provenance,
        "arms": list(ARMS),
        "ladder_cells": [cell_label(k) for k in LADDER_CELLS],
        "refs": agg["refs"],
        "cells": {cell_label(k): {arm: entry for arm, entry in v.items()}
                  for k, v in agg["cells"].items()},
        "scheme_preference_audit": pref,
        "best_vs_best_3bit": best3,
        "best_vs_best_4bit": best4,
        "best_vs_best_3bit_fullgrid": best3_full,
        "july_quant_table": july,
        "crosscheck_published": xck,
        "note": ("best-vs-best margins are point estimates; the paired "
                 "bootstrap over the enlarged selection pool needs per-batch "
                 "records the sweep does not emit (pending separate run)."),
    }
    _atomic_publish_isolated(_REPO_ROOT,
                             out_files["json"].relative_to(_REPO_ROOT),
                             json.dumps(payload, indent=2, allow_nan=False))

    git_note = ("no git state" if provenance["git_sha"] is None else
                f"git {provenance['git_sha'][:9]}"
                f"{' (dirty)' if provenance['git_dirty'] else ''}")
    print(f"[pass1-tables] {args.evidence} evidence; "
          f"{len(agg['cells'])} cells x {len(ARMS)} arms; "
          f"crosscheck {xck['matched']}+{xck['fp32_refs_checked']} exact; "
          f"a1-only-pc-claim={pref['a1_only_claim_holds']}; "
          f"{git_note} -> {out_dir}")


if __name__ == "__main__":
    main()
