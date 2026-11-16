from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tier2_common as tc
from jepa_lm.eval.evaluator import evaluate
from jepa_lm.revision.pass1_tables import (
    EXPECTED_CONTINUATION_SEED,
)
from jepa_lm.revision.sweep_configs import (
    PINNED_N_BATCHES,
    REGISTERED_ARMS,
    REGISTERED_LAYERWISE_OUT_ROOT,
)
from jepa_lm.revision.tier2_measure import (
    PRIMARY_SEED,
    layerwise_matrix,
)
from jepa_lm.systems.kv_quant import KVQuantize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=REGISTERED_ARMS)
    args = parser.parse_args()

    out_rel = (f"{REGISTERED_LAYERWISE_OUT_ROOT}/"
               f"results_s{PRIMARY_SEED}_{args.arm}.json")
    pub = tc.IsolatedPublisher(out_rel, REGISTERED_LAYERWISE_OUT_ROOT)
    try:
        device = torch.device("cuda")
        pkgs = tc.environment_gate(device)
        eval_factory, n_eval_batches, pins = tc.pinned_eval_factory()
        trunk, head, model_path, model_sha = tc.load_registered_model(
            args.arm, PRIMARY_SEED, device)
        matrix = layerwise_matrix()
        print(f"[layerwise] arm={args.arm} model={model_path} "
              f"({len(matrix)} single-layer cells + fp32 ref, "
              f"{n_eval_batches} batches/eval)")
    except BaseException:
        pub.release()
        raise

    provenance = {
        "arm": args.arm, "source_seed": PRIMARY_SEED,
        "continuation_seed": EXPECTED_CONTINUATION_SEED[args.arm],
        "eval_seed": tc.EVAL_SEED,
    }
    rows: list[dict] = []

    def run_eval(quantize_ctx) -> float:
        torch.manual_seed(tc.EVAL_SEED)
        counter = tc.CountingIter(eval_factory())
        with torch.no_grad():
            if quantize_ctx is None:
                metrics = evaluate(trunk, head, counter, device=device)
            else:
                with quantize_ctx:
                    metrics = evaluate(trunk, head, counter, device=device)
        if counter.count != n_eval_batches:
            raise SystemExit(f"[layerwise] short eval pass: "
                             f"{counter.count} != {n_eval_batches}")
        return float(metrics["mean_nll"])

    t0 = time.time()
    reference_nll = run_eval(None)
    rows.append({**provenance, "layer": None, "target": None,
                 "config": "fp32-ref", "nll": reference_nll,
                 "nll_delta_vs_ref": 0.0, "n_batches": n_eval_batches})
    print(f"[layerwise] fp32-ref nll={reference_nll:.6f} "
          f"({time.time()-t0:.1f}s)")

    for i, cell in enumerate(layerwise_matrix()):
        t0 = time.time()
        ctx = KVQuantize(trunk, bits=cell["bits"],
                         granularity=cell["granularity"],
                         targets=cell["target"], mode=cell["mode"],
                         layers=[cell["layer"]])
        nll = run_eval(ctx)
        rows.append({**provenance, "layer": cell["layer"],
                     "target": cell["target"], "config": cell["config"],
                     "nll": nll,
                     "nll_delta_vs_ref": nll - reference_nll,
                     "n_batches": n_eval_batches})
        print(f"[layerwise] {i+1:2d}/48 {cell['config']} L{cell['layer']} "
              f"nll={nll:.6f} d={nll-reference_nll:+.6f} "
              f"({time.time()-t0:.1f}s)")

    if len(rows) != 49:
        raise SystemExit(f"[layerwise] {len(rows)} rows != 49")
    keys = {(r["config"], r["layer"], r["target"]) for r in rows}
    if len(keys) != 49:
        raise SystemExit("[layerwise] duplicate (config, layer, target)")
    for r in rows:
        if r["n_batches"] != PINNED_N_BATCHES:
            raise SystemExit("[layerwise] bad n_batches row")
        if not math.isfinite(r["nll"]):
            raise SystemExit("[layerwise] non-finite nll")
        if abs(r["nll_delta_vs_ref"] - (r["nll"] - reference_nll)) > 0:
            raise SystemExit("[layerwise] delta identity broken")
        if r["config"] == "fp32-ref":
            if r["layer"] is not None or r["target"] is not None:
                raise SystemExit("[layerwise] ref row layer/target must "
                                 "be null")
        elif type(r["layer"]) is not int or r["target"] not in ("k", "v"):
            raise SystemExit("[layerwise] bad layer/target types")

    envelope = {
        "schema_version": "tier2-layerwise-v1",
        "git": pub.git,
        "command": sys.argv,
        "dev_mode": False,
        "env": {**tc.env_record(device), **pins,
                "model_sha256": model_sha,
                "model": str(model_path.relative_to(_REPO_ROOT)),
                "packages_verified": bool(pkgs)},
        "rows": rows,
    }
    pub.publish(envelope)
    print(f"[layerwise] published {out_rel} ({len(rows)} rows)")
    tc.finish()


if __name__ == "__main__":
    main()
