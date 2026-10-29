# Final scientific audit

Branch `review/final-audit`, audit date 2026-10-28. Baseline (before any change): 185 tests passed, the PDF built, 14 stored result files present. Sections A–G follow the audit brief; "Not done" items are listed explicitly in the last section.

## Findings

| # | Finding | Severity | Evidence | Resolution |
|---|---------|----------|----------|------------|
| A1 | The betting confidence sequence was evaluated on a grid but the retained hull was not padded: a true mean lying between two grid points could be excluded because its lower neighbour was. The guarantee was stated as exact. | Medium (numerical, size ≤ 5e-4) | `src/agentaudit/sequential.py` (old `m[idx[0]]`, `m[idx[-1]]`) | Hull widened by one grid cell, documented as a numerical safeguard, not a proof off-grid; `log_capital` exposed; tests added. Affected simulations (E5, S4) rerun. |
| A2 | Paper text said false equivalence occurred "in at most 0.2% of runs … as the theory requires". 0.2% was 1/500 replications, i.e. a simulation frequency with large relative error, presented next to a theoretical statement. The decision rule and its error statement were not written out. | High (claim/wording) | `results/sim_seqm.json` (old, 500 reps) | Section 3.7 now states the capital process, predictability, Ville, the three decision rules and the exact guarantee (any wrong decision of any kind ≤ α, via one event). Simulation rerun at 2000 reps with Clopper–Pearson limits; separate bound/frequency language. |
| A3 | Equivalence power depends strongly on the dispersion of task differences (86% in the main simulation, 9% in an independent three-point-law simulation at margin 10 and true gap 0). | Medium (generalisation) | `results/audit_equivalence.json` | Both reported; text no longer implies a general rate. |
| B1 | "Majority-outcome McNemar … conservative secondary test" is unjustified. It tests a different estimand (majority-success difference 7.8 points vs mean difference 10.2), uses an arbitrary tie rule (2/4 = fail), and is *less* significant (p=0.16, 0.096 with ties dropped) while McNemar on pass^4 is *more* significant (p=0.003). | High (claim) | `results/audit_taubench.json` | Claim removed; majority test presented as a sensitivity analysis of a different estimand; 14-procedure sensitivity table (Appendix F). |
| B2 | Headline numbers verified: retail 10.2 [3.9, 16.5], sign-flip p=0.002, Holm 0.005; airline 3.5 [−6.0, 13.0], majority McNemar p=0.16. Airline upper bootstrap endpoint is 13.5 with a 20 000-draw bootstrap (seed dependence ≤ 0.5 pt). | Low | Same file | Primary analysis declared (studentised sign-flip, task bootstrap, Holm over domains); retail detection holds for every procedure targeting the mean-score estimand. |
| B3 | Janssen (1997) treats the two-sample problem; its use for the one-sample studentised sign-flip was stated as established. | Low | Reference check | Reworded as "by analogy", checked by simulation (E1). |
| B4 | Sample-size figures (95 retail, 787 airline) are single plug-in draws; bootstrap intervals 38–591 and 59–11 215. | Medium (presentation) | existing `reanalysis_taubench.json` | Described as orders of magnitude, not predictions. |
| C1 | "Coverage verified" (public log folders + `no_generation` = 500) ignores `no_logs`: 36 of the 110 "verified" entries list instances under `no_logs` (up to 90), which are *unobserved* outcomes scored as unresolved by the official convention. | Medium | `results/audit_swebench.json` | Added complete-log subset (88 entries), pairwise complete-case and least-favourable-filling bounds. Share of separable pairs 70.8–72.3% across all; headline robust. |
| C2 | No provenance identifier for the SWE-bench snapshot. | Medium | — | Latest commit touching `evaluation/verified` is `92c5a4f` (2026-09-01, GitHub API, queried 2026-10-28); files downloaded 2026-10-27 (mtime). Recorded in paper and reports. The tau-bench source commit was not re-queried. |
| C3 | Inclusion criterion (≥ 50 resolved) excludes only 6 of 135 entries with a results file; claims are about publicly submitted leaderboard rows. | Low | — | Stated in the text. |
| D1 | Simulation proportions reported without Monte Carlo error; replications range 400–2000. | Medium | `results/sim_*.json` | MC s.e. table (Appendix F) and inline s.e. for the two headline numbers (68.3% ± 1.0, 17.9% ± 0.9). |
| D2 | Recommendation "task-level intervals when R>1 or T≥50" overstated: task bootstrap covers 91.6–95.5% in those cells. | Medium | `results/sim_covrob.json` | Grid stated, no threshold derived, mild optimism reported. |
| E1 | Variance components: expected mean squares, estimators, difference variance, G coefficient, `optimal_runs` verified by derivation and Monte Carlo; no defect found. Truncation at zero upward-biases variances (conservative for planning). `optimal_runs` returns a 10^6 sentinel when s_t=0. | Info | `tests/test_variance_audit.py` | Tests: unbiasedness before truncation (3000 reps), difference variance vs Monte Carlo (20 000 reps), brute-force integer optimum over 48 parameter combinations, boundaries. |
| F1 | Local-model raw outcomes, trajectories and cache are not in the repository; Ollama is not installed here. Levels 1, 2, 5 of reproduction are impossible. | High (limitation) | repo contents | Reproducibility statements rewritten by level (Appendix G). Not repeated. |
| F2 | Zero-score control (`qwen2.5-3b-q2`): from stored pairwise p-values, excluding it leaves 1 of 15 pairs not separable (1 of 21 with it); variance decomposition and ranking without it cannot be recomputed. | Low | `results/real_main.json` | Reported with the limitation. |
| G1 | Real-data ablation reference ranking is the full sample (not independent); significant pairs are selected by the same data. | Medium | `scripts/ablate_real.py` | Caption and text rewritten as internal stability; E6 simulation cited for known-truth recovery. Split-sample analysis not possible without per-run outcomes. |
| L1 | Related work: the CS is WSR (2024) unchanged; claims narrowed. Five 2026 references and HAL verified against arXiv / PMLR / ICLR pages. | Low | see below | Related work edited. |

## What was verified for the sequential procedure (A)

1. *Nonnegativity.* Every factor 1 ± λ(X−m) ≥ 1−cap = 1/2 (λ ≤ cap/m resp. cap/(1−m)). Finite log-capital tested for all-0, all-1 and alternating streams.
2. *Predictability.* λ_i depends on X_1..X_{i−1}; test: replacing X_30.. leaves the CS for the first 30 tasks bit-identical.
3. *Martingale.* Exact enumeration over all 3^6 paths of three-point laws with mean m gives E[K_6(m)] = 1 (to 1e-9) against an independently coded reference (plain Python loop agrees with the vectorised capital to 1e-10).
4. *Ville.* Applied to K_t(m) for the true m; the CS is the hull of {m : K_s(m) < 1/α ∀ s ≤ t}, a superset of the exact CS.
5. *Grid.* One-cell padding; grid 401 vs 4001 differ by ≤ 2/400 (test).
6. *Hedging.* Average of two martingales is a martingale; threshold 1/α. Valid.
7. *Decisions.* a/b/equivalent rules each require Δ ∉ CS at the stopping time, so the probability of any wrong decision is ≤ α (not 2α), optional stopping included, under i.i.d. bounded tasks. The point estimate in the band, or one boundary excluded, does not suffice for "equivalent": the whole sequence must lie in [−ε, ε]. Closed band: |Δ| = ε is *inside* the equivalence region.
8. *Simulation (2000 reps, three-point law, margins 0/3/6/10, Δ ∈ {0, ±ε, ε+2, 2ε+2, −(ε+2)} points).* Largest wrong-decision rate 2.70% (upper 95% CP 3.51%); largest CS miss rate 2.90%; largest false-equivalence rate 0.55% (upper 0.98%). Conservative: no violation of 5%. Equivalence at margin 10, Δ=0 declared in 9.2% (this law) vs 85.8% (E5 law).
9. *Assumptions not verifiable by simulation:* tasks i.i.d. from one distribution (a fixed pool in random order is sampling without replacement).

Margins 0.03 and 0.06 reach equivalence essentially never within 300 tasks in either simulation; margin 0.10 only for low-dispersion differences.

## Not done / remaining limitations

* **Not rerun (as of the first audit; S0 coverage/type-I/design/ranking were rerun in the closure pass below):** S0 coverage/type-I/design/ranking simulations, E1, E2, E3, E6 (stored results unchanged; only E5 and S4 were rerun after the sequential fix). The BCa, two-stage and beta-binomial profile-likelihood implementations were not independently re-derived in this audit; existing unit tests cover them.
* **Local-model study** not repeated (no Ollama, no raw data). Split-sample ranking analysis not possible.
* **References:** the five 2026 references and HAL were checked against arXiv/PMLR/ICLR pages (all consistent); the other ~33 entries were checked for internal consistency only (authors, year, venue as I know them), not individually against a publisher page.
* **Estimand caveat:** "population" inference over tasks treats a fixed benchmark as an i.i.d. sample of tasks; tau-bench and SWE-bench tasks are fixed sets.
* tau-bench source commit was not recorded or re-queried.
* The SWE-bench snapshot identity rests on file mtimes and the absence of later commits on that path.

## Readiness

**Conditionally ready** — see `FINAL_CHANGELOG.md`.

---

# Closure pass (A2, 2026-10-28)

* **GitHub:** `origin/review/final-audit` existed at `1539dfe` (verified with `git ls-remote`), `origin/main` at `ed87206`, no pull request for the branch. The user's "GitHub does not appear updated" was because no PR had been opened; the branch itself was pushed. A PR is opened in this pass (see `PR_SUMMARY.md`).
* **S0 contradiction resolved by execution:** S0 is five components. Four (coverage, type-I, design, ranking) were rerun and all 1 136 numeric fields are bit-identical to the first-version results (details in `FINAL_REPRODUCIBILITY_REPORT.md`). The fifth, sequential (S4, `sim_sequential.py`), was **not** in that chain; it was regenerated after the grid fix in the first audit (values changed) and rerun again on 2026-10-29, matching the current file bit for bit (45 fields) but not the first-version file. Execution-evidence table and claim table: `CLAIM_AUDIT.md`. The earlier session's rerun rests on commit history only.
* **BCa / two-stage / profile likelihood:** `tests/test_interval_audit.py` (11 tests): task rows intact under task resampling; two-stage variance equals task-bootstrap variance + within-task variance/(TR) (40 000 draws, 3%); runs-only variance equals W/(TR); BCa agrees with `scipy.stats.bootstrap` (±0.015) and handles all-0/all-1 and highly discrete samples; likelihood equals scipy `betabinom` up to the binomial coefficient; profile interval reduces to the binomial likelihood interval at one run per task and behaves at boundary data; wider under overdispersion. Not proved: validity of the profile interval for arbitrary task heterogeneity (one extra simulation, `audit_bbmix.py`: coverage 94.6% beta, 95.4% two-point mixture, 96.0% point mass, MC s.e. ≈ 1 pt). Manuscript qualified. Code finding: `bootstrap_ci(method="bca")` has no explicit guard for a single task (not changed).
* **Sequential wording:** the 2.70% / 0.55% figures are described as simulation frequencies for the stated three-point-law scenarios with Clopper–Pearson limits; "conservative" is stated as an observation in these scenarios; validity rests on the argument in Section 3.7.
* **tau-bench primary method:** retained; manuscript now states that the studentised sign-flip is not exact under the weak null (asymptotic, supported by E1 with T ∈ {20, 50, 100, 300}, rejection ≤ 5.1%) and that observed skewness is 0.11/0.29; the choice was made from estimand and design and written in the script before the comparison table, but not pre-registered externally.
* **SWE-bench:** Table 7 denominators re-derived from `audit_swebench.py`; definitions of documented failure, absent log, unobserved outcome and verified coverage added; provenance explicitly *inferred*.
* **Local model:** searched the repository, git history (all refs), tags, releases (wheel/sdist/PDF only), Actions artifacts (Pages only) and the local filesystem; no raw outcomes, cache, manifests or logs exist. Study kept as a limited case study; stated in the abstract.
* **References:** all 39 entries looked up (arXiv API / Crossref / DOI); unverifiable against a record: Dror et al. 2018, Holm 1979, Kish, Cronbach et al., Brennan, Efron & Tibshirani (classical, left unchanged). No metadata was altered.
* **Figure fix:** tau-bench figure legend no longer overlaps the curves.
* **Not done:** E1–E3 and E6 not rerun; mypy/CI not run locally; the single-task BCa guard.

Readiness: **Conditionally ready** (unchanged).
