# Revision changelog (branch `paper/major-scientific-revision`)

Base: `main` at the state of commit `daddb2a` ("paper: recompiled PDF without email or affiliation"). Findings
are numbered as in `SCIENTIFIC_AUDIT.md`.

## Claims corrected, weakened or removed

| First version | Revised version | Finding |
|---|---|---|
| Raw trajectories and caches are "released as release assets" | Not published; only the aggregate is stored; local-model numbers cannot be recomputed from raw runs | 1 |
| None of the 128 adjacent SWE-bench positions is distinguishable | 0/128 overall, 1/109 coverage-verified, 3/40 single-attempt, 2/36 both; "almost never" | 2 |
| SWE-bench entries are one run per task, all evaluated on 500 instances | 41 single-attempt, 80 multi-attempt, 8 unknown; 110 coverage-verified, 17 unverifiable, 2 incomplete | 3, 4 |
| Controlled fault injection; every model faces the same failures | Injector implemented but not enabled in the main run; matched probability and seeds, not identical events | 5 |
| More tasks beat more repeats at a fixed budget | Conditional on agent-by-task interaction (ratio 0.96-1.04 at sd 0.2, 1.54-1.56 at sd 1.0) and on an uncapped task pool | 6 |
| Sign-flip test is exact | Exact only under symmetric differences; Monte Carlo beyond 14 non-zero differences; checked against a skewed weak null | 7 |
| Task bootstrap stays near nominal | Not for T = 20, R = 1, extreme prevalence (81-82 %) | 8 |
| tau-bench tasks-needed figures (787 / 95) | Reported with bootstrap intervals (59-11 215 / 38-591); task-level McNemar (p = 0.16) reported next to the sign-flip result (p = 0.0025; Holm 0.005) | 12 |
| q2 model is a degenerate control | Kept, but its score is labelled protocol failure (1200/1200 `no_action`); backend errors (49, 0.6 %) reported | 13 |
| Unverifiable statements (Ollama 0.40.8 defect, hardware) | Marked as unverified author observations | 16 |
| "coverage above 90 %, mypy --strict" | 96.8 % measured; mypy has 4 stub-related errors | 17 |

## Added

- **Research questions** RQ1-RQ5 and a terminology paragraph; limitations on estimand, fixed agents and task
  dependence.
- **Experiments** E1, E2, E3, E5, E6 (`scripts/sim_extra.py`, `docs/experiments.md`, written before the runs) and
  the SWE-bench coverage/attempts audit with four subsets (`scripts/fetch_swebench_meta.py`,
  `scripts/reanalyze_swebench.py`); bootstrap of tasks-needed and Holm over domains for tau-bench.
- **Library**: `studentized_sign_flip`; corrected docstring of `sign_flip_permutation`; adjacent-position counts
  under Holm over all pairs in `reanalysis.py`.
- **Tests**: `tests/test_audit.py` (21: exact unbiasedness, edge cases, Holm/BH vs statsmodels, sign-flip
  enumeration and validity, ANOVA expected mean squares, confidence-sequence validity, margins) and
  `tests/test_manuscript.py` (7: manuscript macros vs `results/`, macro definitions, sensitivity table, link
  style).
- **Figures and tables** (all generated): `asymmetric_null`, `allocation`, `coverage_robustness`; Table 1 (SWE-bench
  sensitivity); legible tick labels; subtle layout fixes.
- **Traceability**: `TRACEABILITY.md` (186 macros -> asset script -> result file -> experiment script), generated
  by `scripts/make_traceability.py`.
- **References**: Janssen 1997, Romano 1990, Benjamini-Yekutieli 2001, Dietterich 1998, Ramdas et al. 2023,
  Lakens 2017, Kish 1965; arXiv ids for SWE-bench and AgentBench; complete author list for Bouthillier et al.
  Existence, title and authors of the arXiv-listed references were checked against the arXiv API.
- **Documents**: `SCIENTIFIC_AUDIT.md`, `EXPERIMENTS_AND_REPRODUCIBILITY.md`, this file.

## Changed (presentation)

- Hyperlinks without green/red boxes (`hidelinks`); consistent terms (task, run, benchmark score, pass@k,
  pass^k; "attempt" only as the SWE-bench field name); the limitations/ethics/conclusion moved to
  `sections/limits.tex`; Appendix E (local-model table) verified populated; reproducibility appendix rewritten with
  verification levels; Makefile targets for the new experiments and the SWE-bench metadata fetch.
- The paper grew from 15 to 20 pages (TMLR style kept, `preprint` option); no venue adaptation was made.

## Unchanged on purpose

The author line, title, benchmark design, library API (other than the additions above), the local-model data and
all original simulation results (re-run, identical).

## Not done

- No re-run of the local-model experiment (no raw data, no Ollama here).
- No fix of the four mypy stub errors; no run of Docker, mkdocs or the CI matrix.
- No new real-agent data (for example other tau-bench models or HAL runs); no analysis with task dependence
  (clustered tasks) in the simulations.
- The GitHub release `v0.2.0` still carries the old PDF; releases, preprints and submissions were not touched.
