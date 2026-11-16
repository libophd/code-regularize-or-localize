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
from jepa_lm.revision.pass1_tables import (
    EXPECTED_CONTINUATION_SEED,
    SEEDS,
)
from jepa_lm.revision.sweep_configs import (
    PINNED_N_BATCHES,
    REGISTERED_ARMS,
    REGISTERED_RECON_OUT_ROOT,
)
from jepa_lm.revision.tier2_measure import (
    N_LAYERS,
    ReconAccumulator,
    both_target_recon_cells,
)
from jepa_lm.systems.kv_capture import KVCapture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=REGISTERED_ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    args = parser.parse_args()

    out_rel = (f"{REGISTERED_RECON_OUT_ROOT}/"
               f"results_s{args.seed}_{args.arm}.json")
    pub = tc.IsolatedPublisher(out_rel, REGISTERED_RECON_OUT_ROOT)
    try:
        device = torch.device("cuda")
        pkgs = tc.environment_gate(device)
        eval_factory, n_eval_batches, pins = tc.pinned_eval_factory()
        trunk, _head, model_path, model_sha = tc.load_registered_model(
            args.arm, args.seed, device)
        cells = both_target_recon_cells()
        print(f"[recon] arm={args.arm} s{args.seed} "
              f"({len(cells)} cells x {N_LAYERS} layers x 2 tensors, "
              f"{n_eval_batches} batches)")
    except BaseException:
        pub.release()
        raise

    acc = ReconAccumulator()
    torch.manual_seed(tc.EVAL_SEED)
    counter = tc.CountingIter(eval_factory())
    t0 = time.time()
    with torch.no_grad():
        for batch in counter:
            input_ids = batch["input_ids"].to(device)
            with KVCapture(trunk) as cap:
                trunk(input_ids)
                k_layers, v_layers = cap.captured()
                for layer in range(N_LAYERS):
                    for tensor, x in (("k", k_layers[layer]),
                                      ("v", v_layers[layer])):
                        acc.update(layer, tensor, x, cells)
            if counter.count % 10 == 0:
                print(f"[recon] batch {counter.count}/{n_eval_batches} "
                      f"({time.time()-t0:.0f}s)")
    if counter.count != n_eval_batches:
        raise SystemExit(f"[recon] short pass: {counter.count} != "
                         f"{n_eval_batches}")

    provenance = {
        "arm": args.arm, "source_seed": args.seed,
        "continuation_seed": EXPECTED_CONTINUATION_SEED[args.arm],
        "eval_seed": tc.EVAL_SEED,
    }
    rows = []
    for layer in range(N_LAYERS):
        for tensor in ("k", "v"):
            if acc.batches[(layer, tensor)] != counter.count:
                raise SystemExit(f"[recon] batch-count drift "
                                 f"L{layer}/{tensor}")
            for cell in cells:
                rows.append({**provenance, "layer": layer,
                             "tensor": tensor, "config": cell["config"],
                             "rel_frobenius": acc.value(
                                 layer, tensor, cell["config"]),
                             "num_sum": acc.num[(layer, tensor,
                                                 cell["config"])],
                             "den_sum": acc.den[(layer, tensor)],
                             "n_batches": counter.count})
    if len(rows) != N_LAYERS * 2 * len(cells):
        raise SystemExit(f"[recon] {len(rows)} rows != expected")
    keys = {(r["layer"], r["tensor"], r["config"]) for r in rows}
    if len(keys) != len(rows):
        raise SystemExit("[recon] duplicate (layer, tensor, config)")
    for r in rows:
        if r["n_batches"] != PINNED_N_BATCHES \
                or not math.isfinite(r["rel_frobenius"]):
            raise SystemExit("[recon] bad row")

    envelope = {
        "schema_version": "tier2-recon-v1",
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
    print(f"[recon] published {out_rel} ({len(rows)} rows, "
          f"{time.time()-t0:.0f}s)")
    tc.finish()


if __name__ == "__main__":
    main()
