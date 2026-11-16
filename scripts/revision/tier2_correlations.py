from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from jepa_lm.revision.pass1_tables import (
    load_bundle_manifest,
    read_bundle_bytes,
    verify_bundle_closure,
)
from jepa_lm.revision.sweep_configs import (
    REGISTERED_ARMS,
    strict_json_loads,
)

SCHEMES = ("per_token", "per_channel")
TENSORS = ("k", "v")
GEOMETRY_VARS = ("aniso_paircos", "top1_share", "scale_dispersion",
                 "kurtosis_max")
N_LAYERS = 12

REGISTERED_ANALYSIS_DIR = "outputs/revision/analysis"
BUNDLE_ANALYSIS_DIR = "outputs/revision/analysis_bundle"
_RAW_PREFIX = "outputs/revision/"
_BUNDLED_RESULT = "analysis/tier2_correlations.json"
BUNDLE_AGREEMENT_TOL = 1e-12


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) \
                and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("spearman needs matched n >= 3")
    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        raise ValueError("degenerate rank variance")
    return num / (dx * dy)


def _read(rel: str, digests: dict[str, str]) -> dict:
    path = _REPO_ROOT / rel
    data = path.read_bytes()
    digests[rel] = hashlib.sha256(data).hexdigest()
    return strict_json_loads(data.decode())


def correlations(read) -> list[dict]:
    geometry = read("outputs/revision/geometry_uniform/results_s42.json")["rows"]
    geo = {(r["arm"], r["layer"], r["tensor"]): r for r in geometry}

    results = []
    for arm in REGISTERED_ARMS:
        lw = read(f"outputs/revision/kv_quant/layerwise/"
                  f"results_s42_{arm}.json")["rows"]
        rc = read(f"outputs/revision/kv_quant/recon/"
                  f"results_s42_{arm}.json")["rows"]
        dnll = {(r["config"], r["layer"], r["target"]):
                r["nll_delta_vs_ref"] for r in lw
                if r["config"] != "fp32-ref"}
        recon = {(r["config"], r["layer"], r["tensor"]):
                 r["rel_frobenius"] for r in rc}
        for scheme in SCHEMES:
            for tensor in TENSORS:
                y = [dnll[(f"3bit-{scheme}-{tensor}", ell, tensor)]
                     for ell in range(N_LAYERS)]
                entry = {"arm": arm, "scheme": scheme, "tensor": tensor,
                         "n_layers": N_LAYERS, "spearman": {}}
                x = [recon[(f"3bit-{scheme}-both", ell, tensor)]
                     for ell in range(N_LAYERS)]
                entry["spearman"]["recon_rel_frobenius"] = spearman(x, y)
                for var in GEOMETRY_VARS:
                    xg = [geo[(arm, ell, tensor)][var]
                          for ell in range(N_LAYERS)]
                    entry["spearman"][var] = spearman(xg, y)
                results.append(entry)
    return results


def summary_lines(results: list[dict]) -> list[str]:
    lines = ["# Tier-2 layer-wise Spearman correlations (descriptive)",
             "", f"{len(results)} (arm x scheme x tensor) cells, "
             "12 layers each. rho ranges below are min..max across the "
             "seven arms.", ""]
    for scheme in SCHEMES:
        for tensor in TENSORS:
            lines.append(f"## scheme={scheme} tensor={tensor.upper()}")
            for var in ("recon_rel_frobenius",) + GEOMETRY_VARS:
                vals = [r["spearman"][var] for r in results
                        if r["scheme"] == scheme
                        and r["tensor"] == tensor]
                lines.append(
                    f"- {var:22} rho {min(vals):+.2f}..{max(vals):+.2f} "
                    f"(median {sorted(vals)[len(vals)//2]:+.2f})")
            lines.append("")
    return lines


def largest_difference(computed: list[dict], registered: list[dict]) -> float:
    if len(computed) != len(registered):
        raise SystemExit("recomputed and registered results differ in size")
    worst = 0.0
    for new, old in zip(computed, registered):
        if ({k: v for k, v in new.items() if k != "spearman"}
                != {k: v for k, v in old.items() if k != "spearman"}
                or set(new["spearman"]) != set(old["spearman"])):
            raise SystemExit(f"recomputed cell {new['arm']}/{new['scheme']}/"
                             f"{new['tensor']} does not match the registered "
                             "cell")
        for var, value in new["spearman"].items():
            worst = max(worst, abs(value - old["spearman"][var]))
    return worst


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evidence", choices=("full", "bundle"), default="full",
                   help="full (default): the published raw outputs under "
                        "outputs/revision/. bundle: their curated copies "
                        "in the artifact bundle.")
    p.add_argument("--bundle-root", default=None,
                   help="bundle mode: root of the artifact bundle "
                        "(default paper/artifacts)")
    args = p.parse_args(argv)
    bundle_mode = args.evidence == "bundle"
    if args.bundle_root is not None and not bundle_mode:
        p.error("--bundle-root is only meaningful with --evidence bundle")

    digests: dict[str, str] = {}
    out = {
        "schema_version": "tier2-correlations-v1",
        "note": ("Descriptive Spearman rank correlations over 12 layers, "
                 "within arm x scheme x tensor; correlated statistics — "
                 "cannot establish mediation. Geometry values from the "
                 "uniform protocol only."),
        "inputs_sha256": digests,
    }
    if bundle_mode:
        bundle_root = Path(args.bundle_root if args.bundle_root is not None
                           else "paper/artifacts")
        if not bundle_root.is_absolute():
            bundle_root = _REPO_ROOT / bundle_root
        manifest, _ = load_bundle_manifest(_REPO_ROOT, bundle_root)
        verify_bundle_closure(bundle_root, manifest)

        def read(rel: str) -> dict:
            bundled = rel[len(_RAW_PREFIX):]
            data = read_bundle_bytes(bundle_root, bundled, manifest)
            digests[bundled] = hashlib.sha256(data).hexdigest()
            return strict_json_loads(data.decode())

        results = correlations(read)
        registered = strict_json_loads(read_bundle_bytes(
            bundle_root, _BUNDLED_RESULT, manifest).decode())["results"]
        worst = largest_difference(results, registered)
        if worst > BUNDLE_AGREEMENT_TOL:
            raise SystemExit(
                f"correlations recomputed from the bundle differ from the "
                f"registered values by up to {worst:.3g} "
                f"(tolerance {BUNDLE_AGREEMENT_TOL:g})")
        out["evidence"] = "bundle"
        out["raw_chain_replayed"] = False
        out["registered_results"] = {"path": _BUNDLED_RESULT,
                                     "max_abs_difference": worst}
        out_dir = _REPO_ROOT / BUNDLE_ANALYSIS_DIR
    else:
        results = correlations(lambda rel: _read(rel, digests))
        out_dir = _REPO_ROOT / REGISTERED_ANALYSIS_DIR
    out["results"] = results

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tier2_correlations.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n")
    lines = summary_lines(results)
    (out_dir / "tier2_correlations.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"written: {out_dir}/tier2_correlations.{{json,md}} "
          f"({len(results)} cells, {len(digests)} inputs digest-recorded)")
    if bundle_mode:
        print(f"agreement with the registered values in the bundle: "
              f"largest absolute difference {worst:.3g}")


if __name__ == "__main__":
    main()
