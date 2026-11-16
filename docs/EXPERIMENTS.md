# Experiment entry points

The package contains the implementations used for the study, together with frozen experiment contracts. The complete historical input tree and checkpoint binaries are not included. The entry points below make the code navigable; listing a driver is not a claim that its original registered run can start from an empty checkout.

## Method map

| Component | Implementation |
|---|---|
| SIGReg and per-position SIGReg | `src/jepa_lm/training/sigreg.py` |
| Training objective and trainer | `src/jepa_lm/training/losses.py`, `trainer.py` |
| Llama-style trunk | `src/jepa_lm/trunks/llama.py` |
| Direct cache regularization (A1) | `src/jepa_lm/systems/a1_probe.py` |
| Frozen-trunk probes (A2/A3) | `src/jepa_lm/systems/bridge_probe.py` |
| CE-only and single-tensor continuation arms | `src/jepa_lm/revision/continuation.py` |
| LoRA adapters and probe authentication | `src/jepa_lm/systems/lora.py`, `src/jepa_lm/revision/lora_probe.py` |
| Simulated cache quantization | `src/jepa_lm/systems/kv_quant.py` |
| Layer-wise measurements | `src/jepa_lm/revision/tier2_measure.py`, `geometry_uniform.py` |

The quantizers simulate perturbations to cache activations. They are not a deployed low-bit cache storage kernel or a throughput benchmark.

## Training and evaluation drivers

Read the command-line options before starting a run:

```bash
python scripts/phase1_real_run.py --help
python scripts/phase1_5_bridge_probe.py --help
python scripts/kv_quant_eval.py --help
python scripts/revision/run_continuation_arm.py --help
python scripts/revision/run_lora_probe.py --help
```

| Driver | Role | Additional inputs |
|---|---|---|
| `phase1_real_run.py` | CE baseline and hidden-state SIGReg pretraining | FineWeb stream and tokenizer access |
| `phase1_5_bridge_probe.py` | A1/A2/A3 probes from a source model | A compatible source checkpoint; data/tokenizer access |
| `kv_quant_eval.py` | Generic quantization evaluation | A compatible checkpoint or saved adapted trunk |
| `revision/run_continuation_arm.py` | Matched continuation dissection | Registered source checkpoint and historical launch prerequisites |
| `revision/run_lora_probe.py` | Exploratory LoRA-scale continuation | Registered source checkpoint and historical launch prerequisites |
| `revision/preflight.py`, `kv_quant_sweep.py`, `kv_quant_sweep_merge.py` | Registered locality-grid chain | Original model registry, freezes, pilot/receipt chain, and input layout |
| `revision/tier2_*.py` | Geometry, reconstruction, layer-wise damage, attention, and correlations | Model/raw measurement inputs described by the frozen contracts |

The shell chains `scripts/revision/run_pass1.sh` and `run_tier2.sh` preserve the original registered workflow. They expect its model paths, local environment, and output layout; they are **not portable setup scripts**. Do not launch them as a quick-start command.

## Recipes and data

The paper describes the training budget, architecture, seeds, evaluation window, and scope of each comparison. Curated `config.json`, `command.txt`, result metadata, and frozen constants provide additional recipe information. Public dataset/tokenizer identities and their recorded revision pins are preserved in the evidence where they were recorded. Earlier pretraining artifacts have the reproducibility limitations stated in the manuscript.

The code streams FineWeb and uses a 32,000-token Llama-compatible tokenizer by default. Data and tokenizer weights are not bundled. Check their upstream access conditions and terms before downloading; the MIT License on the original code does not replace those terms. A synthetic test is not a meaningful experimental result, and an off-recipe run must not be relabeled as one of the paper's registered measurements.

## Regenerating tables versus new measurements

To regenerate the supported tables **without model binaries**, run:

```bash
python scripts/paper_tables.py --artifacts
python scripts/revision/pass1_tables.py --evidence bundle
python scripts/revision/tier2_correlations.py --evidence bundle
```

[Reproducibility](REPRODUCIBILITY.md) lists which tables each command covers and which are not regenerated.

For new measurements, choose an output location that does not overwrite the curated result files, record the actual package versions and data revisions, and preserve command/config/result metadata. Generic training and evaluation code may be used with new checkpoints; the original authenticated drivers intentionally reject a changed registry or provenance chain.

The release cannot yet be described as fully reproducible experimental reruns from a clean checkout: checkpoint download locations and the raw replay dependencies are not provided. See [Reproducibility](REPRODUCIBILITY.md) for the precise supported scope.
