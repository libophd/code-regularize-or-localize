#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=.venv-anchor/bin/python
RECEIPT=outputs/revision/logs/pilot_crosscheck_receipt.json
PILOT_OUT=outputs/revision/kv_quant/groupsweep/results_s42_l_aux__st3.json
LOG=outputs/revision/logs/pass1_progress.log
say() { echo "[pass1 $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
fail() { say "FATAL: $*"; exit 1; }

MODELS='l_aux|42|outputs/phase1_seedconfirm_s42/checkpoints/arm0/baseline/checkpoint_step610352.pt|-
l_aux|43|outputs/phase1_seedconfirm_s43/checkpoints/arm0/baseline/checkpoint_step610352.pt|-
l_aux|44|outputs/phase1_seedconfirm_s44/checkpoints/arm0/baseline/checkpoint_step610352.pt|-
l_aux+l_sigreg|42|outputs/phase1_seedconfirm_s42/checkpoints/arm1/sigreg_l0.01/checkpoint_step610352.pt|-
l_aux+l_sigreg|43|outputs/phase1_seedconfirm_s43/checkpoints/arm1/sigreg_l0.01/checkpoint_step610352.pt|-
l_aux+l_sigreg|44|outputs/phase1_seedconfirm_s44/checkpoints/arm1/sigreg_l0.01/checkpoint_step610352.pt|-
a1_adapted|42|outputs/phase1_5_replicates/s42_saved/a1/adapted_trunk.pt|42
a1_adapted|43|outputs/phase1_5_replicates/s43_saved/a1/adapted_trunk.pt|42
a1_adapted|44|outputs/phase1_5_replicates/s44_saved/a1/adapted_trunk.pt|42
ce_continue|42|outputs/phase1_5_replicates/s42_ce_continue/a1/adapted_trunk.pt|42
ce_continue|43|outputs/phase1_5_replicates/s43_ce_continue/a1/adapted_trunk.pt|42
ce_continue|44|outputs/phase1_5_replicates/s44_ce_continue/a1/adapted_trunk.pt|42
ce_only|42|outputs/revision/phase1_5_replicates/s42_ce_only/a1/adapted_trunk.pt|42
ce_only|43|outputs/revision/phase1_5_replicates/s43_ce_only/a1/adapted_trunk.pt|42
ce_only|44|outputs/revision/phase1_5_replicates/s44_ce_only/a1/adapted_trunk.pt|42
h_k_only|42|outputs/revision/phase1_5_replicates/s42_h_k_only/a1/adapted_trunk.pt|42
h_k_only|43|outputs/revision/phase1_5_replicates/s43_h_k_only/a1/adapted_trunk.pt|42
h_k_only|44|outputs/revision/phase1_5_replicates/s44_h_k_only/a1/adapted_trunk.pt|42
h_v_only|42|outputs/revision/phase1_5_replicates/s42_h_v_only/a1/adapted_trunk.pt|42
h_v_only|43|outputs/revision/phase1_5_replicates/s43_h_v_only/a1/adapted_trunk.pt|42
h_v_only|44|outputs/revision/phase1_5_replicates/s44_h_v_only/a1/adapted_trunk.pt|42'

say "=== PASS-1 CHAIN START at HEAD $(git rev-parse --short HEAD) ==="

if [[ ! -f "$PILOT_OUT" ]]; then
    say "pilot bootstrap launching (l_aux/42/st3)"
    PYTHONPATH=src $PY scripts/revision/preflight.py \
        --model outputs/phase1_seedconfirm_s42/checkpoints/arm0/baseline/checkpoint_step610352.pt \
        --arm l_aux --source-seed 42 --sweep st3 \
        --out-json "$PILOT_OUT" \
        --manifest-out outputs/revision/logs/pilot_manifest.json \
        --pilot-bootstrap --launch >> "$LOG" 2>&1 \
        || fail "pilot launch failed — see $LOG"
fi
[[ -f "$PILOT_OUT" ]] || fail "pilot result absent after launch"
say "pilot result present"

if [[ ! -f "$RECEIPT" ]]; then
    say "minting crosscheck receipt (comparability gate)"
    PYTHONPATH=src $PY scripts/revision/pilot_crosscheck.py \
        --pilot "$PILOT_OUT" --arm l_aux --source-seed 42 \
        --receipt-out "$RECEIPT" >> "$LOG" 2>&1 \
        || fail "COMPARABILITY GATE FAILED — no pass-1 data may be retained; see $LOG"
fi
say "receipt present — queueing receipt-mode launches"

TOTAL=0; DONE=0
while IFS='|' read -r arm seed path cont; do
    [[ -z "$arm" ]] && continue
    for sweep in st3 bt3 bt4; do
        [[ "$arm|$seed|$sweep" == "l_aux|42|st3" ]] && continue
        TOTAL=$((TOTAL + 1))
        out="outputs/revision/kv_quant/groupsweep/results_s${seed}_${arm}__${sweep}.json"
        if [[ -f "$out" ]]; then
            say "skip (exists): $(basename "$out")"; DONE=$((DONE + 1)); continue
        fi
        mani="outputs/revision/logs/manifest_s${seed}_${arm}__${sweep}_$(date -u +%Y%m%dT%H%M%SZ).json"
        args=(--model "$path" --arm "$arm" --source-seed "$seed" --sweep "$sweep"
              --out-json "$out" --manifest-out "$mani"
              --crosscheck-receipt "$RECEIPT" --launch)
        [[ "$cont" != "-" ]] && args+=(--continuation-seed "$cont")
        say "LAUNCH $arm s$seed $sweep"
        PYTHONPATH=src $PY scripts/revision/preflight.py "${args[@]}" >> "$LOG" 2>&1 \
            || fail "launch failed: $arm s$seed $sweep — see $LOG"
        [[ -f "$out" ]] || fail "no output after launch: $out"
        DONE=$((DONE + 1))
        say "DONE   $arm s$seed $sweep ($DONE/$((TOTAL)) so far)"
    done
done <<< "$MODELS"

say "=== PASS-1 COMPLETE: pilot + $DONE receipt-mode results present ==="
