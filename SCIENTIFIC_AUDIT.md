# Scientific audit of "AgentAudit: How Much of an Agent Leaderboard Is Signal?"

Scope: implementation, mathematics, statistics, data, results, references and presentation of the first
version of the paper (`paper/agentaudit-paper.pdf` at commit `daddb2a`, 15 pages) and of the repository.
Branch of the revision: `paper/major-scientific-revision`. Every finding states what was checked, what
was found and what was done. "Reproduced" means re-run from the scripts in this session.

## 1. Method

1. Baseline: the existing 157 tests passed; the five original simulation scripts were re-run in full and
   compared value by value with the stored `results/*.json` (872 numeric values, 0 differences), so the
   original simulation numbers are reproducible bit for bit.
2. Source audit of `src/agentaudit/*` against the formulas in the paper (metrics, intervals, variance,
   compare, ranking, sequential, power, simulate, reanalysis, ingest, failures).
3. New tests with an independent oracle (`tests/test_audit.py`, 21 tests) and manuscript-vs-results
   consistency tests (`tests/test_manuscript.py`, 7 tests).
4. Experiments pre-specified in `docs/experiments.md` before they were run (E1, E2, E3, E5, E6, S1).
5. Audit of the public data used by the re-analyses (SWE-bench `metadata.yaml`, S3 log prefixes).
6. Reference check against the arXiv API (existence, title, authors) and publisher information; manual
   check of venue, volume and pages for the older journal references.
7. Rewrite of the manuscript, with every number generated from `results/` (see `TRACEABILITY.md`).

## 2. Findings

Severity: **A** changes a central claim or the validity of a result; **B** weakens or conditions a claim;
**C** wording, presentation or reproducibility hygiene.

| # | Sev. | Finding | Evidence | Resolution |
|---|---|---|---|---|
| 1 | A | The first version said raw trajectories and record/replay caches are "released as release assets". They are not in the repository, and the only GitHub release (`v0.2.0`) holds the wheel, the source archive and the old paper PDF, not trajectories. | repo listing, `.gitignore`, `gh release view v0.2.0` | Claim removed. The paper and `docs/reproducing.md` now say that only the aggregate `results/real_main.json` is stored and that the local-model numbers cannot be recomputed from raw runs. |
| 2 | A | The abstract/section 2 claim that **none** of the 128 adjacent SWE-bench positions is distinguishable is true only for the full set of 129 entries. With coverage-verified entries it is 1 of 109, with single-attempt entries 3 of 40, with both 2 of 36 (Holm over all pairs of the subset). | `results/reanalysis_swebench.json`, `subsets` | Claim restated as "almost never"; Table 1 reports all four subsets. |
| 3 | A | "One run per task / per system" for SWE-bench is not supported: the `attempts` field of `metadata.yaml` is 1 for only 41 entries, greater than 1 for 80 (best-of-n or multi-sample systems) and missing for 8. | `scripts/fetch_swebench_meta.py`, `reanalyze_swebench.py` | Stated in the protocol; sensitivity analysis on single-attempt entries; the comparison is described as one of leaderboard rows, not of models. |
| 4 | A | The first version assumed that every entry was evaluated on all 500 instances. Coverage could be verified (public S3 log folders plus `no_generation` equal to 500 instances) for 110 entries; 17 have no public logs and 2 are incomplete. | same | Assumption removed; sensitivity analysis on coverage-verified entries. The share of separable pairs is stable (71.7 % overall and coverage-verified; 69.3 % single-attempt). |
| 5 | A | The paper did not state that the main local-model run used no fault injection (`--fault-rate` defaults to 0), while presenting controlled stochasticity as a feature. Moreover the injector draws sequentially along the call sequence, so two agents face the same failure *pattern* only if they call tools in the same order; "every model faces the same failures" was overstated. | `scripts/run_real.py`, `bench` injector | Both statements corrected in section 4 and the protocol. |
| 6 | B | "More tasks beat more repeats at a fixed budget" was stated unconditionally. It follows from `(R s_t + s_e)/B` and holds only if the agent-by-task interaction is non-zero. Experiment E2 (9 configurations, 1500 replications): SE ratio of R=10 vs R=1 is 0.96-1.04 at interaction sd 0.2 and 1.54-1.56 at sd 1.0; task sd hardly matters. A capped task pool changes the advice. | `results/sim_alloc.json` | Claim made conditional; figure 6 added. |
| 7 | B | Sign-flip test described as "exact". It is exact (conditionally on `|d|`) only under the strong symmetric null, and Monte Carlo beyond 14 non-zero differences. E1 (skewed paired differences with mean zero, skewness up to 5.5): maximum rejection 5.4 % (sign-flip), 5.1 % (studentised), 6.7 % (paired bootstrap, paired t); no method exceeded the pre-specified 2-point tolerance at T >= 100. | `results/sim_asym.json` | Docstrings and text corrected; studentised sign-flip added to the library; the default is kept. |
| 8 | B | Task bootstrap was presented as near nominal everywhere. E3 (1000 replications): 91.6-95.5 % for T >= 50 and R > 1, but 81-82 % with T = 20, R = 1, extreme prevalence and homogeneous tasks (BCa from 80 %). Pooled Wilson is valid at R = 1. | `results/sim_covrob.json` | Recommendation made conditional. |
| 9 | B | Pooled Wilson interval: coverage 68.3 % at T = 100, R = 10 (original configuration) and 68-76 % at task sd 1.5, R = 10; roughly nominal when tasks are nearly homogeneous. The original "wrong by almost thirty points" holds for the original configuration only. | `sim_coverage.json`, `sim_covrob.json` | Discussion reworded ("about twenty-five points when tasks differ"). |
| 10 | B | Equivalence/non-significance. Sequential "equivalent" requires a margin. E5: with margin 0 it is never declared; with a 10-point margin and a true gap of 0 it is declared in 86 % of runs; when the true gap exceeds the margin it is declared in at most 0.2 % (below alpha). With identical agents and margin 0 the sequence stays undecided in 98 % of runs. | `results/sim_seqm.json` | Section on what "not distinguishable" means added; text never equates it with equivalence. |
| 11 | B | Ranking and near ties. E6: the tied pair is rejected in at most 2.0 % of replications (Holm is conservative and controls the family-wise error); a pair 2.7 points apart is separated in 19 % of replications even with T = 500 and R = 3 (0 % with T = 50). | `results/sim_rankpw.json` | Added to results and discussion. |
| 12 | B | tau-bench: the retail gap is detected by the task-level sign-flip test (p = 0.0025; 0.005 after Holm over the two domains) but not by the exact task-level McNemar test on per-task majority outcomes (p = 0.16). Plug-in "tasks needed" figures are very uncertain: bootstrap 95 % interval 59-11 215 tasks (airline, point estimate 787) and 38-591 (retail, 95). The model label is Claude 3.5 Sonnet (new, 2024-10-22). | `results/reanalysis_taubench.json` | All reported; plug-in values flagged as orders of magnitude. |
| 13 | B | `qwen2.5-3b-q2`: 100 % of its 1200 runs end with `no_action`; its score of 0 measures protocol failure, not capability. 49 runs (0.6 %) are `backend_error` and scored as failures. | `results/real_main.json` | Kept as a degenerate control, separated from capability in the text, backend errors reported. |
| 14 | B | The seven agents are a fixed set, not a sample from a population of agents; the "agent" variance share is a descriptive spread. | model assumptions | Stated in the setting and in the local-model section. |
| 15 | B | Benjamini-Hochberg is valid under independence or PRDS; pairwise tests sharing agents are dependent. | theory | Holm is the default; BH reported as secondary with the caveat and the Benjamini-Yekutieli reference. |
| 16 | B | Local-model experiment cannot be repeated here (no Ollama models, no raw trajectories). The Ollama defect (Appendix B) and the hardware description are statements by the author that cannot be verified from the repository. | environment | Reported as unverified observations; not re-run. |
| 17 | C | Reproducibility checklist asserted "coverage above 90 %, `mypy --strict`". Measured: coverage 96.8 % (gate 90 %); `ruff` clean after formatting; `mypy --strict` reports four stub-related errors with the installed numpy/SciPy stubs (`results.py:89`, `forms.py:158`, `intervals.py:93,163`) that do not affect behaviour; the project's CI (`mypy src`, Python 3.11 configuration) passes. | `pytest --cov`, `mypy --python-version 3.14 src` | Checklist replaced by a verification-level statement; errors not changed (outside the scientific scope). |
| 18 | C | Green and red hyperref boxes; inconsistent use of "attempt", "run", "trial"; Appendix E reported as empty. | PDF | `hidelinks`; terminology defined in the Introduction (task, run, benchmark score, pass@k, pass^k; "attempts" kept only as the SWE-bench field name); Appendix E (local-model table) verified populated in the new PDF. |
| 19 | C | Reference details: SWE-bench and AgentBench lacked arXiv ids; Bouthillier et al. had an incomplete author list. | arXiv API | Fixed; seven related-work and method references added (Janssen 1997, Romano 1990, Benjamini-Yekutieli 2001, Dietterich 1998, Ramdas et al. 2023, Lakens 2017, Kish 1965). |
| 20 | C | A statement of mine in an intermediate progress message gave the pooled-Wilson coverage under heterogeneity as "55-75 %"; the correct range is 68-76 % (55-65 % is the runs-only bootstrap). | `results/sim_covrob.json` | The paper uses the correct values. |

## 3. What was verified and found correct

- pass@k and pass^k estimators are exactly unbiased for n = 1..8 (exact binomial sums) and behave correctly at
  the edges (k = 1, k = n, c = 0, c = n); k > n raises.
- Holm and Benjamini-Hochberg match `statsmodels.multipletests` on random inputs.
- The enumerated sign-flip p-value equals full enumeration; the Monte Carlo version agrees with it; its type-I
  error under a symmetric null is at most nominal.
- The ANOVA expected mean squares recover the true components in simulation (600 balanced datasets, 12 %
  tolerance); with R = 1 the interaction is flagged as confounded; truncation keeps the raw values.
- The betting confidence sequence does not exclude the true mean more often than alpha (300 sequences); equivalence
  cannot be declared with margin 0.
- Analytic and simulated standard errors agree within 3.8 % (original design) and 4.6 % (E2).
- The original simulations reproduce exactly (see 1).

## 4. Honest assessment

The core statistical message (cluster by task; pooling runs breaks coverage and test validity; plan with the
variance decomposition; report what public data do not document) is sound and now supported by experiments
that were able to contradict it, and several did contradict parts of the first version (items 2, 6, 8, 9). The
paper is a methodology and audit study with a simulation core; its empirical leg on real agents is thin: two
frontier models on tau-bench, public SWE-bench rows of mixed provenance, and seven small quantised models whose
raw data are not retained. Results about local models do not transfer to frontier agents. The simulations use
one generator family (logit-normal item response); robustness along prevalence, heterogeneity, skewness,
margins and near ties was examined, but not misspecified dependence between tasks. Tasks in real benchmarks are
likely correlated, which would make task-level intervals optimistic. Confidence sequences are conservative in
the settings tested and the equivalence claim needs a margin that a practitioner must justify.
