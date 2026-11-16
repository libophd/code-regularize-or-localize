PYTHON ?= python
export PYTHONPATH := src
export OMP_NUM_THREADS := 2
export MKL_NUM_THREADS := 2

.PHONY: smoke test tables figures

smoke:
	CUDA_VISIBLE_DEVICES="" $(PYTHON) scripts/phase1_dry_run.py

test:
	CUDA_VISIBLE_DEVICES="" $(PYTHON) -m pytest -q

tables:
	$(PYTHON) scripts/paper_tables.py --artifacts
	$(PYTHON) scripts/revision/pass1_tables.py --evidence bundle
	$(PYTHON) scripts/revision/tier2_correlations.py --evidence bundle

figures:
	$(PYTHON) scripts/paper_figures.py --artifacts paper/artifacts --out-dir outputs/figures
