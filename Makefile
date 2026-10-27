ifeq ($(OS),Windows_NT)
VENV_BIN := .venv/Scripts
BOOT ?= py -3
else
VENV_BIN := .venv/bin
BOOT ?= python3
endif
PY ?= $(VENV_BIN)/python
.PHONY: install lint type test cov reproduce docs clean sim public real paper
install:
	$(BOOT) -m venv .venv && $(PY) -m pip install --upgrade pip && $(PY) -m pip install -e ".[dev,plots,docs]"
lint:
	$(VENV_BIN)/ruff format --check src tests scripts web && $(VENV_BIN)/ruff check src tests scripts web
type:
	$(VENV_BIN)/mypy src
test:
	$(VENV_BIN)/pytest
cov:
	$(VENV_BIN)/pytest --cov=agentaudit --cov-report=term-missing --cov-fail-under=90
docs:
	$(VENV_BIN)/mkdocs build --strict

# ---- reproduction of every number in the paper -------------------------------------------------
sim:
	$(PY) scripts/sim_coverage.py --reps 2000 --n-boot 1000
	$(PY) scripts/sim_tests.py --reps 2000
	$(PY) scripts/sim_design.py
	$(PY) scripts/sim_sequential.py
	$(PY) scripts/sim_ranking.py

public:
	$(PY) scripts/fetch_swebench.py && $(PY) scripts/reanalyze_swebench.py
	$(PY) scripts/fetch_taubench.py && $(PY) scripts/reanalyze_taubench.py

# Local-model runs are long (hours) and need Ollama; they are resumable. See docs/reproducing.md.
real:
	OLLAMA_NUM_PARALLEL=4 $(PY) scripts/run_real.py --models llama3.2-1b qwen2.5-1.5b qwen2.5-3b llama3.2-3b qwen2.5-3b-q8 qwen2.5-3b-q2 qwen2.5-7b --runs 10 --workers 4 --condition main
	$(PY) scripts/analyze_real.py --dir runs/real/main
	$(PY) scripts/ablate_real.py --dir runs/real/main

paper:
	$(PY) scripts/make_paper_assets.py
	cd latex && tectonic main.tex

# Everything that needs no model: simulations, public re-analysis, figures, paper.
reproduce: sim public paper
