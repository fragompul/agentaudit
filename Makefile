PY ?= .venv/bin/python
.PHONY: install lint type test cov reproduce docs clean sim public real paper
install:
	uv venv --python 3.12 .venv && uv pip install -e ".[dev,plots,docs]"
lint:
	.venv/bin/ruff format --check src tests scripts && .venv/bin/ruff check src tests scripts
type:
	.venv/bin/mypy src
test:
	.venv/bin/pytest
cov:
	.venv/bin/pytest --cov=agentaudit --cov-report=term-missing --cov-fail-under=90
docs:
	.venv/bin/mkdocs build --strict

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
