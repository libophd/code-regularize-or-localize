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

The locality grid ran one model and one sweep set per invocation. `revision/preflight.py --launch` checks the launch conditions, writes the launch manifest, and starts `revision/kv_quant_sweep.py`. After the first run, `revision/pilot_crosscheck.py` compares it with the earlier per-scheme results and writes the comparability receipt that every later launch requires. `revision/kv_quant_sweep_merge.py` then merges the three sweep sets of each model. Each merged result file records, for its three source runs, the command line and the launch manifest. These drivers expect the original model paths, local environment, and output layout; they are **not portable setup scripts**. Do not launch them as a quick-start command.

## Script index

Every script in this repository, what it does, and the result of the paper it produced. Paths in the last column are relative to `paper/artifacts/`; table and figure numbers are those of the published paper. The generators, the dry run, and the analysis step of the bootstrap run from this repository alone; the other scripts need inputs that are not in it (checkpoints, complete run directories, dataset and tokenizer access).

### `scripts/`

| Script | What it does | Result |
|---|---|---|
| `phase1_real_run.py` | Pretrains and evaluates the CE baseline and the CE + λ·SIGReg(h) models | `phase1_exitgate/`, `phase1_seedconfirm_s*/` (Tables 2 and 5, Figure 4) |
| `phase1_consolidate.py` | Checks and merges the per-model result files of one pretraining sweep | `phase1_exitgate/sweep_metrics.json` (Table 5, Figure 4) |
| `phase1_seedconfirm_aggregate.py` | Computes the across-seed statistics and within-seed differences of the paired runs, and the seed-42 comparison between the two software stacks | `phase1_seedconfirm_aggregate.json` (Table 2) |
| `export_hf_checkpoint.py` | Converts a checkpoint to the Hugging Face format and checks that the converted model gives the same logits | `hf_export/`; the converted models are the input of the zero-shot evaluation (Tables 6 and 7, run with the external lm-evaluation-harness) and of the STS-B probe |
| `sts_isotropy_probe.py` | Measures the STS-B correlation of sentence embeddings, with and without All-but-the-Top post-processing | `sts_probe/` (Section 5.3) |
| `phase1_5_bridge_probe.py` | Runs one of the probes A1, A2, or A3 | `phase1_5/`, `phase1_5_replicates/` (Section 5.4, Figure 3) |
| `kv_quant_eval.py` | Evaluates simulated cache quantization over one set of schemes (`--config-set`) | `kv_quant/results_*.json`, `kv_quant/controls_*.json` (Figure 2, Tables 4 and 10) |
| `kv_quant_bootstrap.py` | Paired bootstrap over packed batches for the best-configuration margins: `run` records the per-batch losses, `analyze` computes the intervals from those records | `kv_quant/bootstrap_records.json`, `kv_quant/bootstrap_summary.json` (Section 5.5) |
| `kv_channel_stats.py` | Collects per-channel statistics of the keys and values of each layer | `channel_stats/results_all.json` (Section 7) |
| `paper_tables.py` | Table generator | Tables 2, 4, 5, and 10; see [Reproducibility](REPRODUCIBILITY.md) |
| `paper_figures.py` | Figure generator | Figures 2-4 |
| `phase1_dry_run.py` | Trains and evaluates a tiny model on synthetic data | None (quick start) |

### `scripts/revision/`

The experiments added when the paper was revised: the locality grid, the matched continuation runs, the LoRA-scale probe, and the layer-wise diagnostics.

| Script | What it does | Result |
|---|---|---|
| `run_continuation_arm.py` | Runs one CE-only, h+K, or h+V continuation | `continuations/s*_{ce_only,h_k_only,h_v_only}/` (rows of Table 3, Section 5.4) |
| `run_lora_probe.py` | Runs the LoRA-scale probe | `probes/lora_kv/s42/` (Table 8) |
| `preflight.py` | Checks the launch conditions of one locality-grid run, writes its launch manifest, and starts the run | The launch manifests recorded in `kv_quant/groupsweep/` |
| `kv_quant_sweep.py` | Evaluates one model on one of the three sweep sets of the locality grid | The source runs of `kv_quant/groupsweep/` |
| `pilot_crosscheck.py` | Compares the first run with the earlier per-scheme results and writes the comparability receipt | `kv_quant/groupsweep/pilot_crosscheck_receipt.json` (Appendix E) |
| `kv_quant_sweep_merge.py` | Checks and merges the three sweep sets of one model | `kv_quant/groupsweep/results_s*__merged.json` (Tables 3 and 9) |
| `pass1_tables.py` | Table generator for the locality grid | Tables 3, 4, and 9; see [Reproducibility](REPRODUCIBILITY.md) |
| `tier2_layerwise.py` | Quantizes one layer at a time and measures the loss increase | `kv_quant/layerwise/` (Appendix C) |
| `tier2_recon.py` | Measures the per-layer reconstruction error of quantized keys and values | `kv_quant/recon/` (Appendix C) |
| `tier2_attention.py` | Measures attention distortion under quantized keys with full-precision queries | `kv_quant/attention/` (Table 13) |
| `tier2_geometry.py` | Collects per-layer geometry statistics of keys and values under one protocol | `geometry_uniform/results_s42.json` (Appendix C) |
| `tier2_correlations.py` | Table generator: computes the layer-wise rank correlations from the three kinds of layer-wise result files | `analysis/tier2_correlations.json` (Table 12); see [Reproducibility](REPRODUCIBILITY.md) |
| `tier2_common.py` | Shared setup of the four measurement scripts above | None |

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
