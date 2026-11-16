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
)
from jepa_lm.revision.sweep_configs import (
    PINNED_N_BATCHES,
    REGISTERED_ARMS,
    REGISTERED_ATTENTION_OUT_ROOT,
)
from jepa_lm.revision.tier2_measure import (
    ATTENTION_CELLS,
    N_LAYERS,
    PRIMARY_SEED,
    attention_layer_sums,
)
from jepa_lm.systems.kv_capture import KVCapture


def _rope_config(trunk) -> tuple[int, int, float]:
    cfg = trunk.model.config
    scaling = getattr(cfg, "rope_scaling", None)
    rope_type = (scaling or {}).get("rope_type", "default")
    if rope_type != "default":
        raise NotImplementedError(f"rope_type {rope_type!r} unsupported")
    if getattr(cfg, "partial_rotary_factor", 1.0) != 1.0:
        raise NotImplementedError("partial rotary unsupported")
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    if n_heads % n_kv:
        raise ValueError("n_heads not divisible by n_kv")
    return n_heads, n_kv, float(getattr(cfg, "rope_theta", 10000.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=REGISTERED_ARMS)
    args = parser.parse_args()
    head_chunk = 2

    out_rel = (f"{REGISTERED_ATTENTION_OUT_ROOT}/"
               f"results_s{PRIMARY_SEED}_{args.arm}.json")
    pub = tc.IsolatedPublisher(out_rel, REGISTERED_ATTENTION_OUT_ROOT)
    try:
        device = torch.device("cuda")
        pkgs = tc.environment_gate(device)
        eval_factory, n_eval_batches, pins = tc.pinned_eval_factory()
        trunk, _head, model_path, model_sha = tc.load_registered_model(
            args.arm, PRIMARY_SEED, device)
        n_heads, n_kv, rope_base = _rope_config(trunk)
        print(f"[attention] arm={args.arm} ({len(ATTENTION_CELLS)} cells "
              f"x {N_LAYERS} layers, {n_eval_batches} batches, "
              f"heads={n_heads}/kv={n_kv}/base={rope_base})")
    except BaseException:
        pub.release()
        raise

    sums: dict[tuple[int, str], dict[str, float]] = {}
    torch.manual_seed(tc.EVAL_SEED)
    counter = tc.CountingIter(eval_factory())
    t0 = time.time()
    with torch.no_grad():
        for batch in counter:
            input_ids = batch["input_ids"].to(device)
            with KVCapture(trunk, capture_q=True) as cap:
                trunk(input_ids)
                k_layers, _v = cap.captured()
                q_layers = cap.captured_q()
            for layer in range(N_LAYERS):
                got = attention_layer_sums(
                    q_layers[layer], k_layers[layer], ATTENTION_CELLS,
                    n_heads=n_heads, n_kv=n_kv, rope_base=rope_base,
                    head_chunk=head_chunk)
                for name, parts in got.items():
                    key = (layer, name)
                    acc = sums.setdefault(key, {
                        "logit_num": 0.0, "logit_den": 0.0,
                        "map_num": 0.0, "map_den": 0.0})
                    for field, value in parts.items():
                        acc[field] += value
            if counter.count % 10 == 0:
                print(f"[attention] batch {counter.count}/"
                      f"{n_eval_batches} ({time.time()-t0:.0f}s)")
    if counter.count != n_eval_batches:
        raise SystemExit(f"[attention] short pass: {counter.count}")

    locus_by_cell = {c["cell"]: c["locus"] for c in ATTENTION_CELLS}
    provenance = {
        "arm": args.arm, "source_seed": PRIMARY_SEED,
        "continuation_seed": EXPECTED_CONTINUATION_SEED[args.arm],
        "eval_seed": tc.EVAL_SEED,
    }
    rows = []
    for (layer, cell_name), acc in sorted(sums.items()):
        for side in ("logit", "map"):
            if acc[f"{side}_den"] <= 0.0 \
                    or not math.isfinite(acc[f"{side}_den"]):
                raise SystemExit(f"[attention] degenerate {side} "
                                 f"denominator L{layer}/{cell_name}")
        rows.append({
            **provenance, "layer": layer, "cell": cell_name,
            "locus": locus_by_cell[cell_name],
            "logit_rel_frob": math.sqrt(acc["logit_num"]
                                        / acc["logit_den"]),
            "map_rel_frob": math.sqrt(acc["map_num"] / acc["map_den"]),
            "logit_num_sum": acc["logit_num"],
            "logit_den_sum": acc["logit_den"],
            "map_num_sum": acc["map_num"],
            "map_den_sum": acc["map_den"],
            "n_batches": counter.count,
        })
    if len(rows) != N_LAYERS * len(ATTENTION_CELLS):
        raise SystemExit(f"[attention] {len(rows)} rows != expected")
    keys = {(r["layer"], r["cell"]) for r in rows}
    if len(keys) != len(rows):
        raise SystemExit("[attention] duplicate (layer, cell)")
    for r in rows:
        if r["n_batches"] != PINNED_N_BATCHES \
                or not math.isfinite(r["logit_rel_frob"]) \
                or not math.isfinite(r["map_rel_frob"]):
            raise SystemExit("[attention] bad row")

    envelope = {
        "schema_version": "tier2-attention-v1",
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
    print(f"[attention] published {out_rel} ({len(rows)} rows, "
          f"{time.time()-t0:.0f}s)")
    tc.finish()


if __name__ == "__main__":
    main()
