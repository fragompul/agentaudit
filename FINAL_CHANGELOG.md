# Final changelog (scientific claims and results that changed)

Previous values are kept in git history (`main`, commit `ed87206`); nothing was replaced silently.

| Claim / result | Before | After | Why |
|---|---|---|---|
| Confidence-sequence guarantee | exact on a 2001-point grid | grid hull widened by one cell; off-grid guarantee called numerical | grid not padded (A1) |
| Sequential equivalence error | "at most 0.2% … as the theory requires" (1 of 500 replications) | ≤ α = 5% is the theoretical bound for any wrong decision; simulation: ≤ 0.1% per cell in E5 (upper 0.44%, 2000 reps), ≤ 0.55% (upper 0.98%) in an independent three-point law | bound and frequency were conflated (A2) |
| E5 replications | 500 | 2000 | Monte Carlo precision |
| E5 point values | e.g. margin 10, gap 0: 86%; true gap 1.9 pt: 75% | 85.8%, 72.7% (new CS; most within MC error) | grid padding + more reps |
| S4 sequential (the S0 sequential component) | 600 reps; seq power at 7.5 pt 69.5% | 68.8%; null 1.3% (was 1.5%) | grid padding; hence S0 is bit-for-bit reproduced only for its other four components |
| Equivalence at margin 10 pt, gap 0 | 86% (only law tested) | 86% in E5 law; 9% in a more dispersed three-point law | dependence on dispersion (A3) |
| Majority McNemar | "conservative secondary test" | sensitivity analysis of a different estimand; p=0.16 (ties = failure), 0.096 (ties dropped); pass^4 McNemar p=0.003; pass@4 p=0.115 | B1 |
| tau-bench primary analysis | plain sign-flip implied | studentised sign-flip + task bootstrap + Holm (retail p=0.002, Holm 0.005); retail detection holds for every mean-score procedure | declared primary; B2 |
| Airline bootstrap CI | [−6.0, 13.0] | [−6.0, 13.5] with 20 000 draws; both kept, seed sensitivity noted | B2 |
| SWE-bench "coverage verified" | 110 entries, implied complete | 36 of them list `no_logs`; 88 entries have none; share separable 70.8–72.3% under all treatments | C1 |
| Studentised sign-flip validity | "asymptotically valid (Janssen 1997)" | by analogy with Janssen's two-sample theory, supported by E1 | B3 |
| Interval recommendation | task-level intervals for R>1 or T≥50 | conditional on the simulated grid; task bootstrap mildly optimistic (91.6–95.5%) there | D2 |
| Real-data ablation | agreement with full-data ranking as accuracy | internal stability relative to the full sample | G1 |
| Reproducibility | "every number can be regenerated" | aggregates, simulations and public re-analyses regenerate; local-model run cannot be repeated or recomputed from raw outcomes | F1 |
| Related work | CS listed as the method used | CS is WSR (2024) unchanged; contribution is the equivalence rule, audit and simulation | L1 |

Unchanged headline results (S0 coverage, type-I, design and ranking values are bit-identical to the first version; S4 is not, see above): 129 entries, 8 256 pairs, 71.7% separable after Holm, 0 of 128 adjacent positions after Holm (2 unadjusted); tau-bench retail 10.2 points [3.9, 16.5]; coverage 68.3% and run-level McNemar 17.9% (now with MC s.e. 1.0 and 0.9 points); all other S0/E1–E3/E6 simulation values (not rerun).
