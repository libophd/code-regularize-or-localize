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
from jepa_lm.eval.channel_stats import (
    RunningChannelMoments,
    scale_dispersion,
)
from jepa_lm.revision.geometry_uniform import (
    RESERVOIR_CAPACITY,
    RESERVOIR_SEED,
    SeededReservoir,
    mean_pairwise_cosine,
    top1_spectral_share,
)
from jepa_lm.revision.pass1_tables import (
    EXPECTED_CONTINUATION_SEED,
)
from jepa_lm.revision.sweep_configs import (
    PINNED_N_BATCHES,
    REGISTERED_ARMS,
    REGISTERED_GEOMETRY_OUT_ROOT,
)
from jepa_lm.revision.tier2_measure import (
    N_LAYERS,
    PRIMARY_SEED,
)
from jepa_lm.systems.kv_capture import KVCapture


class _StreamingCov:
    def __init__(self, d: int, device: torch.device) -> None:
        self.sum1 = torch.zeros(d, dtype=torch.float64, device=device)
        self.sum2 = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.n = 0

    def update(self, x: torch.Tensor) -> None:
        xd = x.double()
        self.sum1 += xd.sum(dim=0)
        self.sum2 += xd.T @ xd
        self.n += x.shape[0]

    def covariance(self) -> torch.Tensor:
        mu = self.sum1 / self.n
        return self.sum2 / self.n - torch.outer(mu, mu)


def main() -> None:
    argparse.ArgumentParser().parse_args()
    out_rel = f"{REGISTERED_GEOMETRY_OUT_ROOT}/results_s{PRIMARY_SEED}.json"
    pub = tc.IsolatedPublisher(out_rel, REGISTERED_GEOMETRY_OUT_ROOT)
    try:
        device = torch.device("cuda")
        pkgs = tc.environment_gate(device)
        eval_factory, n_eval_batches, pins = tc.pinned_eval_factory()
    except BaseException:
        pub.release()
        raise

    rows = []
    model_shas: dict[str, str] = {}
    for arm in REGISTERED_ARMS:
        try:
            trunk, _head, model_path, model_sha = tc.load_registered_model(
                arm, PRIMARY_SEED, device)
        except BaseException:
            pub.release()
            raise
        model_shas[arm] = model_sha
        acc = {(layer, tensor): {
            "reservoir": SeededReservoir(RESERVOIR_CAPACITY, trunk.d_model
                                         if hasattr(trunk, "d_model")
                                         else 768, seed=RESERVOIR_SEED),
            "cov": None,
            "moments": None,
        } for layer in range(N_LAYERS) for tensor in ("k", "v")}
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
                        flat = x.reshape(-1, x.shape[-1]).float()
                        slot = acc[(layer, tensor)]
                        if slot["cov"] is None:
                            d = flat.shape[-1]
                            slot["cov"] = _StreamingCov(d, device)
                            slot["moments"] = RunningChannelMoments(d)
                            if slot["reservoir"].buffer.shape[1] != d:
                                slot["reservoir"] = SeededReservoir(
                                    RESERVOIR_CAPACITY, d,
                                    seed=RESERVOIR_SEED)
                        slot["cov"].update(flat)
                        slot["moments"].update(flat)
                        slot["reservoir"].update(flat)
                if counter.count % 10 == 0:
                    print(f"[geometry] {arm} batch {counter.count}/"
                          f"{n_eval_batches} ({time.time()-t0:.0f}s)")
        if counter.count != n_eval_batches:
            raise SystemExit(f"[geometry] short pass for {arm}: "
                             f"{counter.count}")
        provenance = {
            "arm": arm, "source_seed": PRIMARY_SEED,
            "continuation_seed": EXPECTED_CONTINUATION_SEED[arm],
            "eval_seed": tc.EVAL_SEED,
        }
        for (layer, tensor), slot in sorted(acc.items()):
            res = slot["reservoir"].collect()
            if res.shape[0] != RESERVOIR_CAPACITY:
                raise SystemExit("[geometry] reservoir underfilled")
            rows.append({
                **provenance, "layer": layer, "tensor": tensor,
                "aniso_paircos": mean_pairwise_cosine(res),
                "top1_share": top1_spectral_share(
                    slot["cov"].covariance()),
                "scale_dispersion": scale_dispersion(
                    slot["moments"].std()),
                "kurtosis_max": float(
                    slot["moments"].excess_kurtosis().max()),
                "n_tokens": slot["moments"].n,
                "reservoir_seed": RESERVOIR_SEED,
            })
        print(f"[geometry] {arm} done ({time.time()-t0:.0f}s)")
        trunk.to("cpu")
        del trunk
        torch.cuda.empty_cache()

    expected = len(REGISTERED_ARMS) * N_LAYERS * 2
    if len(rows) != expected:
        raise SystemExit(f"[geometry] {len(rows)} rows != {expected}")
    keys = {(r["arm"], r["layer"], r["tensor"]) for r in rows}
    if len(keys) != len(rows):
        raise SystemExit("[geometry] duplicate (arm, layer, tensor)")
    n_expected_tokens = PINNED_N_BATCHES * 8 * 2048
    for r in rows:
        if r["n_tokens"] != n_expected_tokens:
            raise SystemExit(f"[geometry] n_tokens {r['n_tokens']} != "
                             f"{n_expected_tokens}")
        for field in ("aniso_paircos", "top1_share", "scale_dispersion",
                      "kurtosis_max"):
            if not math.isfinite(r[field]):
                raise SystemExit(f"[geometry] non-finite {field}")

    envelope = {
        "schema_version": "tier2-geometry-v1",
        "git": pub.git,
        "command": sys.argv,
        "dev_mode": False,
        "env": {**tc.env_record(torch.device("cuda")), **pins,
                "model_sha256_by_arm": model_shas,
                "packages_verified": bool(pkgs)},
        "rows": rows,
    }
    pub.publish(envelope)
    print(f"[geometry] published {out_rel} ({len(rows)} rows)")
    tc.finish()


if __name__ == "__main__":
    main()
