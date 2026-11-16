#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=.venv-anchor/bin/python
LOG=outputs/revision/logs/tier2_progress.log
ARMS="l_aux l_aux+l_sigreg a1_adapted ce_continue ce_only h_k_only h_v_only"

say() { echo "[tier2-chain $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() { say "FAILED: $*"; exit 1; }

run() {
  local label="$1" out="$2"; shift 2
  if [[ -f "$out" ]]; then say "skip $label (exists)"; return 0; fi
  say "start $label"
  PYTHONPATH=src "$@" >>"$LOG" 2>&1 || fail "$label (see $LOG)"
  [[ -f "$out" ]] || fail "$label finished but $out missing"
  say "done $label"
}

for arm in $ARMS; do
  run "attention/$arm" \
      "outputs/revision/kv_quant/attention/results_s42_${arm}.json" \
      $PY scripts/revision/tier2_attention.py --arm "$arm"
done
for arm in $ARMS; do
  run "layerwise/$arm" \
      "outputs/revision/kv_quant/layerwise/results_s42_${arm}.json" \
      $PY scripts/revision/tier2_layerwise.py --arm "$arm"
done
run "geometry" "outputs/revision/geometry_uniform/results_s42.json" \
    $PY scripts/revision/tier2_geometry.py
for arm in $ARMS; do
  for seed in 42 43 44; do
    run "recon/s${seed}_${arm}" \
        "outputs/revision/kv_quant/recon/results_s${seed}_${arm}.json" \
        $PY scripts/revision/tier2_recon.py --arm "$arm" --seed "$seed"
  done
done
say "CHAIN COMPLETE"
