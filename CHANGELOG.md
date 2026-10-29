# Changelog

All notable changes are documented here ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/), [SemVer](https://semver.org/)).

## [0.3.2] - 2026-10-29

### Changed
- Visual polish of the paper (no change to content or results): one figure style (`scripts/paper_style.py`: text-width
  figures, 8/7 pt type, one colour per method or agent across all figures, hairline grids, shared legends), consistent
  table typography and float placement, appendix tables kept with their sections.

## [0.3.1] - 2026-10-29

### Fixed
- `agentaudit.sequential`: the confidence-sequence hull is widened by one grid cell (the guarantee off the grid is
  numerical, not exact); `log_capital` is exposed.

### Added
- Final scientific audit (PR #19): `FINAL_SCIENTIFIC_AUDIT.md`, `FINAL_REPRODUCIBILITY_REPORT.md`, `FINAL_CHANGELOG.md`,
  `FINAL_TRACEABILITY.md`, `CLAIM_AUDIT.md`; audit scripts (`audit_equivalence.py`, `audit_taubench.py`,
  `audit_swebench.py`, `audit_bbmix.py`, `check_refs.py`) and tests for the betting capital process, variance
  components, allocation rule and BCa / two-stage / beta-binomial intervals.

### Changed
- Paper (24 pages): explicit sequential decision rule and error bound, tau-bench sensitivity table, SWE-bench
  missing-log analysis, Monte Carlo error reporting, precise reproducibility statement (S0 four components rerun
  bit for bit; E5, S4 regenerated after the fix; E1-E3, E6 not rerun; local-model raw data unavailable).
- E5 rerun with 2000 replications; S4 regenerated after the grid fix.

## [0.3.0] - 2026-10-27

### Added
- Pre-specified experiments E1-E6 (`scripts/sim_extra.py`, `docs/experiments.md`), SWE-bench coverage/attempts audit,
  `studentized_sign_flip`, `tests/test_audit.py`, `tests/test_manuscript.py`, `TRACEABILITY.md`,
  `SCIENTIFIC_AUDIT.md`, `EXPERIMENTS_AND_REPRODUCIBILITY.md`, `REVISION_CHANGELOG.md`.

### Changed
- Paper rewritten with research questions and corrected claims (see `REVISION_CHANGELOG.md`).
- `sign_flip_permutation` documentation states exactness only under symmetry.

## [0.2.0] - 2026-10-27

### Changed
- New website: editorial landing page with an interactive leaderboard and charts generated from `results/`,
  a redesigned browser playground (drop zone, KPI strip, downloadable report, power planner) and a matching
  MkDocs theme. Fonts are served locally (no Google Fonts).
- HTML report uses the paper and ink palette.
- Makefile works on Windows; GitHub Actions bumped to current majors.
- Playground starts Python on first interaction, so the page stays light; contrast and layout-shift fixes.

## [0.1.0] - 2026-10-23

### Added
- Statistics: unbiased pass@k and pass^k; Wilson, Clopper-Pearson, Beta-Binomial and profile-likelihood
  intervals; task / run / two-stage and BCa bootstrap; crossed variance decomposition with generalizability
  coefficients; paired tests (exact McNemar, cluster bootstrap, sign-flip) with Holm and Benjamini-Hochberg;
  rank-uncertainty distributions; analytic and simulated power planning with optimal task/run allocation;
  anytime-valid betting confidence sequences.
- AgentAudit-Bench: six offline deterministic environments (virtual file system, SQLite, CRM REST, bookings,
  web forms, tabular + calculator), 360 procedurally generated tasks, state-based checkers, seeded fault
  injection, oracle/null self-audit.
- Agents: uniform text tool protocol, Ollama and OpenAI-compatible local backends, synthetic skill agents,
  record/replay cache. Resumable run-major runner.
- Ingestion adapters for tau-bench and SWE-bench; failure taxonomy; Markdown/HTML reports; CLI; web playground.
