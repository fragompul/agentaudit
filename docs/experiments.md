# Pre-specified experiments (scientific revision)

This file was written **before** the experiments in `scripts/sim_extra.py` and the SWE-bench sensitivity
analysis were run. Each entry states the question, the design, and the result that would support or
contradict the claim. Results, including unfavourable ones, are reported in the paper and in
`SCIENTIFIC_AUDIT.md`.

| ID | Question | Design | Claim supported if | Claim contradicted if |
|---|---|---|---|---|
| E1 `asym` | Is the sign-flip test valid when the paired differences are skewed with mean zero (weak null)? | `d_t = a_t - b_t`, task-level win probability gap `+0.45` with probability `q in {0.5, 0.1, 0.03}` and a small compensating loss otherwise (`E[delta] = 0`), `T in {20,50,100,300}`, `R in {1,5}`, 1500 replications, alpha 0.05. Compare sign-flip, studentised sign-flip, paired bootstrap, paired t. | Rejection rate within +-2 MC standard errors (about 1.1 points) of 5% | A method exceeds 5% by more than 2 points at `T >= 100` |
| E2 `alloc` | Does "more tasks beat more repeats" at a fixed attempt budget depend on heterogeneity and interaction, and does a capped task pool change it? | `sigma_task in {0.5,1.5,2.5}` x `sigma_interaction in {0.2,0.6,1.0}`, budget 1200 attempts, `R in {1,2,5,10,20}`, true skill gap 0.25 logit, 1500 replications; analytic variance from a large reference sample; optimal `R*` for task/run cost ratios `{0,1,10,100}`; analytic power with a pool capped at 120 tasks | SE ratio (R=10 vs R=1) is above 1 whenever interaction is non-zero; capped pool reverses the advice | SE ratio is about 1 in some cells (then the claim is conditional on interaction) |
| E3 `covrob` | Is the coverage ranking of interval methods stable across prevalence, task heterogeneity, `T` and `R`? | prevalence (logit -2.5, 0.3, 2.5) x `sigma_task in {0.3,1.5}` x `T in {20,50,100}` x `R in {1,3,10}`, 1000 replications | Task bootstrap within 92-97%, pooled Wilson under-covers when `sigma_task` is large and `R > 1`, and does not when `sigma_task` is small | Task-level methods fail badly at floor/ceiling prevalence (then restrict the recommendation) |
| E5 `seqm` | How do margin and minimum task count change stopping, and is "equivalent" declared when it should not be? | true gap from 0 to about 0.09, margin in `{0,0.03,0.06,0.10}`, `min_tasks in {10,30}`, 300 tasks, `R=3`, 500 replications | `P(equivalent)` when the true gap exceeds the margin stays below alpha | It exceeds alpha |
| E6 `rankpw` | In a leaderboard with one exactly tied pair and close pairs, how often does Holm separate pairs of a given true gap, and is the family-wise error controlled? | six agents, `T in {50,150,500}`, `R in {1,3}`, 400 replications; exact McNemar for `R=1`, paired t otherwise | False rejection of the tied pair at most alpha; separation increases with true gap and `T` | Tied pair rejected more than alpha |
| S1 SWE-bench | Do the conclusions depend on assuming that every entry was evaluated on all 500 instances and ran once? | Verify coverage from the public S3 log prefixes plus `no_generation`; read `attempts` from `metadata.yaml`; repeat the analysis on coverage-verified, single-attempt and both subsets | Fraction distinguishable and adjacent-pair results qualitatively unchanged | They change materially (then state which claim is not robust) |

Monte Carlo precision: a rejection rate near 5% from 1500 replications has standard error 0.56 points; a
coverage near 95% from 1000 replications has 0.69 points. Values are reported with at most that precision.
