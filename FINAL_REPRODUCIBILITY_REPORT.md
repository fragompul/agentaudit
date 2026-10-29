# Final reproducibility report

Environment: Windows 11, Python 3.14.6, 16 logical cores, TeX Live 2025, no GPU, no Ollama. Package installed with `pip install -e . --no-deps`.

## Levels of reproduction, by component

| Component | 1. Raw outcomes from recorded trajectories | 2. Statistics from raw outcomes | 3. Aggregates → numbers | 4. Aggregates → figures/tables | 5. Independent rerun of the data generation |
|---|---|---|---|---|---|
| Simulations S0, E1–E3, E5, E6, S4 | n/a (synthetic, seeded) | Seeded scripts regenerate every cell. **Rerun in the audit:** S0 coverage/type-I/design/ranking (bit-identical to first version, 2026-10-28); S4 and E5 (after the sequential fix; S4 again on 2026-10-29, identical to current file); equivalence study. **Not rerun:** E1, E2, E3, E6 (stored results of the first revision). | yes | yes | n/a |
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
* CI: not run locally. On 2026-10-29 GitHub Actions ran on PR #19 for commit `f47573a` (run 40861310937): docker, docs, lint-and-types, site pass; the `test (3.11/3.12/3.13)` jobs and the overall conclusion `success` per `gh run list`. The final-pass commit is checked after push (see PR).
* `mypy --strict src` locally (Python 3.14): fails before checking with `numpy/__init__.pyi:737: error: Type statement is only supported in Python 3.12 and greater [syntax]`; not claimed to pass locally.

## S0 rerun in the closure pass (2026-10-28)

* Commit: `1539dfe` plus uncommitted wording edits (no change to simulation code). Python 3.14.6, NumPy/SciPy as installed, Windows 11.
* Commands (from the repository root, one chain, log in `audit_logs/s0_rerun.log`):
  `python scripts/sim_coverage.py --reps 2000 --n-boot 1000 --workers 12; python scripts/sim_tests.py --reps 2000 --workers 12; python scripts/sim_design.py; python scripts/sim_ranking.py`
  (seeds fixed in the scripts; the `sim_coverage` step took 812 s).
* Comparison: every numeric field in the regenerated `sim_coverage.json` (292), `sim_tests.json` (595), `sim_design.json` (133) and `sim_ranking.json` (116) compared with the committed file; maximum absolute difference **0.0** (bit-identical), excluding the wall-clock `seconds` field. No tolerance was needed. Regenerated files were then discarded (`git checkout`) so only the committed results remain.
* Scope: this chain covered four of the five S0 components. `sim_sequential.py` (S0d/S4) was not in it (the log contains no output of it).
* S4 rerun on 2026-10-29 (`python scripts/sim_sequential.py`, log `audit_logs/s4_rerun.log`, 2.5 min): all 45 numeric fields identical (max abs diff 0.0) to the committed post-grid-fix `results/sim_sequential.json`; versus the first-version file (`ed87206`) they differ (rates, median tasks), as documented in `FINAL_CHANGELOG.md`.
* Contradiction resolved: the manuscript's "re-run unchanged and reproduce exactly" was previously supported only by commit-history inference; it is now supported by this execution. The first audit report's "S0 not rerun" described that audit session only. E1, E2, E3, E6 were **not** rerun in this pass (stored results only); E5 and S4 were rerun in the first audit.
* Beta-binomial coverage under non-beta mixing (`scripts/audit_bbmix.py`, 500 reps per scenario, 3 workers): profile interval 94.6% (beta), 95.4% (two-point mixture), 96.0% (point mass); task bootstrap 93.8%, 93.6%, 93.4% (MC s.e. about 1 point).
* Reference lookups (`scripts/check_refs.py`): arXiv API for 16 arXiv entries and Crossref for 14 journal/proceedings entries matched author, title and year; Crossref returns the 2023 online date for Waudby-Smith and Ramdas (print year 2024 kept) and does not index Dror et al. (ACL 2018), Holm (1979) and the classical books (Kish, Cronbach, Brennan, Efron and Tibshirani), which were therefore not verified against a record. Card et al. (EMNLP 2020, DOI 10.18653/v1/2020.emnlp-main.745), Lakens (2017, DOI 10.1177/1948550617697177), Efron (1987, DOI 10.1080/01621459.1987.10478410) and Ramdas et al. (2023, DOI 10.1214/23-sts894) were confirmed by DOI.
