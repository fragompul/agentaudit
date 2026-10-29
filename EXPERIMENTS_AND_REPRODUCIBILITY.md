# Experiments and reproducibility

Verification levels used throughout: **R** = reproduced end to end in the revision session (re-run from the
script, compared with the stored result or newly stored); **P** = partially verified; **N** = not rerun.
Environment: Windows 11, Python 3.14, NumPy/SciPy as installed, TeX Live 2025. Commands are run from the
repository root with `PYTHONPATH=src` unless the package is installed (`pip install -e ".[dev,plots]"`).

## 1. Experiments

| ID | Question | Script | Output | Pre-specified in | Level | Wall-clock here |
|---|---|---|---|---|---|---|
| S0a | Interval coverage (RQ1) | `scripts/sim_coverage.py --reps 2000 --n-boot 1000 --workers 6` | `results/sim_coverage.json` | first version | R (identical) | minutes |
| S0b | Type-I error of paired tests (RQ1) | `scripts/sim_tests.py --reps 2000 --workers 6` | `results/sim_tests.json` | first version | R (identical) | minutes |
| S0c | Fixed-budget design (RQ2) | `scripts/sim_design.py` | `results/sim_design.json` | first version | R (identical) | < 1 min |
| S0d | Sequential vs fixed vs peeking (RQ4) | `scripts/sim_sequential.py` | `results/sim_sequential.json` | first version | R (identical) before the sequential grid fix; regenerated after it (values changed, see FINAL_CHANGELOG.md); current script reproduces the current file bit for bit (2026-10-29, `audit_logs/s4_rerun.log`) | minutes |
| S0e | Ranking calibration | `scripts/sim_ranking.py` | `results/sim_ranking.json` | first version | R (identical) | minutes |
| E1 | Weak null with skewed paired differences | `scripts/sim_extra.py asym` | `results/sim_asym.json` | `docs/experiments.md` | R (new) | 90 s |
| E2 | Tasks vs repeats, interaction, cost ratio, capped pool | `scripts/sim_extra.py alloc` | `results/sim_alloc.json` | same | R (new) | 11 s |
| E3 | Coverage robustness (prevalence, heterogeneity, T, R) | `scripts/sim_extra.py covrob` | `results/sim_covrob.json` | same | R (new) | 530 s |
| E5 | Sequential margins and minimum tasks | `scripts/sim_extra.py seqm` | `results/sim_seqm.json` | same | R (new) | 157 s |
| E6 | Pairwise Holm on a near-tied leaderboard | `scripts/sim_extra.py rankpw` | `results/sim_rankpw.json` | same | R (new) | 6 s |
| S1 | SWE-bench sensitivity to coverage and attempts | `scripts/fetch_swebench_meta.py`, `scripts/reanalyze_swebench.py` | `results/reanalysis_swebench.json` | same | R (new) | network + minutes |
| T1 | tau-bench re-analysis (with bootstrap of tasks-needed, Holm over domains) | `scripts/reanalyze_taubench.py` | `results/reanalysis_taubench.json` | extension | R | minutes |
| L1 | Local models, 7 x 120 x 10 | `scripts/run_real.py`, `analyze_real.py`, `ablate_real.py` | `results/real_main.json`, `ablation_main.json` | first version | **N** (stored aggregate only) | hours (author's laptop) |

There is no experiment "E4": the numbering follows `docs/experiments.md`, which lists E1, E2, E3, E5, E6 and S1.

Notes on the runs: the original simulations were run twice because the first attempt with 14 workers failed with
a NumPy memory error (`ArrayMemoryError`, 22.9 MiB); the second used 6 workers and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. E1 failed once because the symmetric control used a jump of 0.45
and produced probabilities outside [0, 1]; the control now uses 0.25 (a deviation from the pre-registered jump,
stated in the paper). No result was selected, dropped or re-run for being unfavourable.

### Results against the pre-specified criteria

| ID | Criterion supporting the claim | Outcome |
|---|---|---|
| E1 | Rejection within +-2 MC standard errors of 5 %; contradicted if a method exceeds 5 % by more than 2 points at T >= 100 | Not contradicted. Maximum 5.4 % (sign-flip), 5.1 % (studentised), 6.7 % (bootstrap, t) over all cells; the 6.7 % cells are 3 MC standard errors above 5 % but within 2 points. Sign-flip tests are conservative on average (3.6 %). |
| E2 | SE ratio above 1 whenever interaction is non-zero | Supported for interaction sd 0.6 and 1.0 (1.26-1.56); **contradicted at interaction sd 0.2** (0.96-1.04): the claim is conditional. |
| E3 | Task bootstrap within 92-97 %; Wilson under-covers when task sd is large and R > 1 and not when small | Task bootstrap 91.6-95.5 % for T >= 50, R > 1 (lower edge slightly under 92); **contradicted** at T = 20, R = 1, extreme prevalence and homogeneous tasks (81-82 %). Wilson: 68-76 % at task sd 1.5, R = 10; 85-93 % at task sd 0.3, R = 10 (partly under-covering). |
| E5 | P(equivalent) when the true gap exceeds the margin stays below alpha | Supported (maximum 0.2 %). |
| E6 | Tied pair rejected at most alpha; separation increases with gap and T | Supported (maximum 2.0 %). |
| S1 | Conclusions qualitatively unchanged | Share of separable pairs unchanged (68.5-71.7 %); the count of separable adjacent positions changes from 0 to 1-3, so the "none" claim is **not robust**. |

## 2. Tests and checks

| Check | Command | Result |
|---|---|---|
| Unit, property and regression tests | `PYTHONPATH=src python -m pytest -q` | 185 passed (157 original, 21 in `tests/test_audit.py`, 7 in `tests/test_manuscript.py`) |
| Coverage | `python -m pytest --cov=agentaudit` | 96.8 % (gate 90 %) |
| Lint and format | `python -m ruff check src tests scripts web && python -m ruff format --check src tests scripts web` | clean |
| Types | `PYTHONPATH=src python -m mypy --python-version 3.14 src` | CI (`mypy src`, Python 3.11 config): **passes**. Local Python 3.14 with the newest numpy stubs: **4 errors** (stub typing: `results.py:89`, `forms.py:158`, `intervals.py:93`, `intervals.py:163`); with the repository setting `python_version = "3.11"` mypy cannot parse the installed NumPy stubs. Not changed. |
| Manuscript consistency | `tests/test_manuscript.py` | macros match results; every macro used in the text is defined; the SWE-bench sensitivity table is part of the paper; `hidelinks` present |
| Original simulations vs stored | re-run and compare | 872 values, 0 differences |
| Paper build | `python scripts/make_paper_assets.py && python scripts/make_traceability.py && cd latex && latexmk -pdf main.tex` | 20 pages, no undefined references or citations |
| CI on the pull request (GitHub Actions) | `gh pr checks` | all jobs pass: tests on Python 3.11, 3.12 and 3.13, lint-and-types, docs, site, docker (not run locally) |

## 3. Reproducing everything that needs no model

```bash
make install            # or: pip install -e ".[dev,plots]"
make test
make reproduce          # sim (original + E1-E6) -> public (SWE-bench meta + tau-bench) -> assets -> PDF
```

Individual pieces:

```bash
python scripts/sim_extra.py covrob              # one of asym, alloc, covrob, seqm, rankpw
python scripts/fetch_swebench_meta.py && python scripts/reanalyze_swebench.py
python scripts/make_paper_assets.py && python scripts/make_traceability.py
cd latex && latexmk -pdf -interaction=nonstopmode main.tex
```

Every random stream is seeded. Floating-point results can differ in the last digits across platforms and library
versions (the tau-bench file changed only in float noise when regenerated). The public re-analyses depend on the
public repositories' state at download time.

## 4. What cannot be reproduced and why

- **Local-model run (L1).** It needs Ollama, the seven model tags in `configs/models.json` (about 15 GB) and
  hours of compute; the raw trajectories and record/replay caches were not retained, so only the stored aggregate
  `results/real_main.json` exists. Its numbers are checked for consistency with the paper, not recomputed.
- **Ollama tool-call defect (Appendix B)** and the hardware description are the author's statements; no log is
  stored.
- **SWE-bench**: the files are those of `swe-bench/experiments` at download time; entries without public logs
  (17) cannot be coverage-verified.

## 5. Data and licenses

tau-bench trajectories: MIT (sierra-research/tau-bench). SWE-bench experiments repository: no license declared, so
only derived aggregates and download scripts are stored. No personal data, no paid API, no cloud job.
