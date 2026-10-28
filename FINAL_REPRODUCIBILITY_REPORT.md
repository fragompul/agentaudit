# Final reproducibility report

Environment: Windows 11, Python 3.14.6, 16 logical cores, TeX Live 2025, no GPU, no Ollama. Package installed with `pip install -e . --no-deps`.

## Levels of reproduction, by component

| Component | 1. Raw outcomes from recorded trajectories | 2. Statistics from raw outcomes | 3. Aggregates → numbers | 4. Aggregates → figures/tables | 5. Independent rerun of the data generation |
|---|---|---|---|---|---|
| Simulations S0, E1–E3, E5, E6, S4 | n/a (synthetic, seeded) | Seeded scripts regenerate every cell. **Rerun in this audit: E5, S4** (after the sequential fix). Others not rerun; stored files unchanged. | yes | yes | n/a |
| Audit A equivalence (new) | n/a | `scripts/audit_equivalence.py 2000` (~25 min, 12 workers) | yes | yes | n/a |
| tau-bench | Public trajectories downloadable (`scripts/fetch_taubench.py`); present in `data/raw/taubench` | **Recomputed in this audit** with `scripts/audit_taubench.py`; all headline values reproduced | yes | yes | not applicable (public data) |
| SWE-bench Verified | Public result files (`fetch_swebench*.py`); present in `data/raw/swebench_*` | **Recomputed** with `scripts/audit_swebench.py` and `reanalyze_swebench.py`; 129 entries / 8 256 pairs / 71.7% Holm / 0 of 128 adjacent Holm reproduced | yes | yes | not applicable |
| Local-model study (7 models, 120 tasks, 10 runs) | **No**: trajectories, per-run outcomes and cache are not in the repository | **No** | yes (`results/real_main.json`) | yes | **No**: no Ollama server here; not attempted |

"Aggregates → numbers/figures/tables" (levels 3–4) is `python scripts/make_paper_assets.py` followed by `latexmk`; it verifies consistency and typesetting only and must not be read as reproduction of the experiments.

## Local-model study: what is and is not possible

* Possible: regenerate Table/Figures from `results/real_main.json` and `results/ablation_main.json`; recompute the Holm count of pairs with and without `qwen2.5-3b-q2` from the stored per-pair p-values (done: 1 of 21 → 1 of 15 pairs not separable).
* Not possible: variance decomposition, rankings, split-half statistics and sub-design ablations without the degenerate control; any statistic needing per-run outcomes.
* A rerun would need: Ollama, the seven tags/quantisations in `configs/models.json`, `scripts/run_real.py --models … --runs 10 --workers 4 --temperature 0.7` (seed 0, 256 tokens per turn). Hardware (stated in the paper as an Apple M5 laptop, 16 GB), Ollama build and model digests were not archived, and server-side nondeterminism means individual outcomes would not match. No pilot was run, and none was estimated beyond "hours on a laptop".

## Data provenance

* SWE-bench Verified: `swe-bench/experiments`, path `evaluation/verified`. Latest commit touching the path (GitHub API, queried 2026-10-28): `92c5a4f1cfe7a81345db140f055dc57b535b2f58`, 2026-09-01. Local files have modification time 2026-10-27 (after that commit), so they are consistent with that state; this is an inference from timestamps, not a recorded hash.
* tau-bench: `sierra-research/tau-bench` historical trajectories; commit not recorded.

## Commands executed in this audit

* `python -m pytest -q` before (185 passed) and after changes (all passed, including new `test_sequential_audit.py`, `test_variance_audit.py`, `test_final_audit_results.py`).
* `ruff check`/`ruff format --check` clean on `src tests scripts`. `mypy --strict` could not run locally (numpy stub syntax requires Python ≥ 3.12 semantics; same limitation noted in the paper); not run.
* `scripts/audit_equivalence.py 2000`, `scripts/audit_taubench.py`, `scripts/audit_swebench.py`, `scripts/sim_extra.py seqm` (2000 reps/cell), `scripts/sim_sequential.py`.
* `python scripts/make_paper_assets.py`; `latexmk -pdf main.tex` from a clean aux state: 23 pages, no undefined references, no overfull boxes; PDF pages 10, 20–23 inspected as images.
* CI (docs, site, Docker, Python matrix) was not run.
