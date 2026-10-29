# Claim and execution audit (closure pass)

Legend. **Type**: T theoretical, S simulated, E empirical (public data), L local-model aggregate, A assumption. **Fresh**: re-executed from scripts/raw data in the audit sessions (A1 = first audit, A2 = this closure pass); "stored" = only the committed aggregate was used.

## 1. Execution evidence

| Experiment | Claimed | Evidence found | Verdict |
|---|---|---|---|
| S0 (coverage, type-I, design, ranking) "rerun unchanged, reproduce stored results exactly" (`protocol.tex`, also `EXPERIMENTS_AND_REPRODUCIBILITY.md` level R) | rerun before release 0.3.0 | Commit `489fa3b` (2026-10-27 22:46) changed `sim_coverage.json`, `sim_design.json`, `sim_tests.json`: diffs are only last-digit float differences (e.g. 0.5515276537459712 → 0.551527653745971; analytic power changes at 1e-15; profile-likelihood coverage mean width at 5e-11), rates unchanged. That pattern is consistent with a real rerun, but **no execution log exists**; file mtimes are checkout times, not run times. The A1 report ("S0 not rerun") was wrong about what the earlier session had done: A1 neither inspected this nor ran S0. | Earlier rerun *plausible, undocumented*. **Rerun in A2** (`audit_logs/s0_rerun.log`), see section 2 for the comparison. |
| E1 asym, E2 alloc, E3 covrob, E5 seqm, E6 rankpw | run once in the earlier session | commits `489fa3b`, `72732d7` add scripts and results; no logs | Executed (commit evidence). Not rerun in A1/A2, except E5 (A1, 2000 reps). |
| S0d / S4 sequential (`sim_sequential.py`) | not in the 2026-10-28 S0 chain (`audit_logs/s0_rerun.log` has coverage, tests, design, ranking output only) | rerun 2026-10-29 (`audit_logs/s4_rerun.log`): 45 numeric fields bit-identical to the current (post-fix) file, max abs diff 16.0 (median tasks) and non-zero rate differences vs the first-version file | Reproduces the current file only; **not** identical to the first-version values because of the grid fix. |
| E5 seqm, S4 sequential | rerun after grid fix | A1: commit `1539dfe` changes both result files with new values; E5 reps 500→2000 | Executed in A1 (evidence: result diffs; run output not retained). |
| Audit A equivalence (`audit_equivalence.py`) | run, 2000 reps | commit `78a49b2` | Executed A1. |
| tau-bench, SWE-bench audit scripts | run | commit `78a49b2`; outputs reproduced stored headline values | Executed A1. |
| Beta-binomial coverage under non-beta mixing (`audit_bbmix.py`) | new in A2 | `audit_logs/bbmix.log`, `results/audit_bbmix.json` | Executed A2. |
| Interval unit tests (BCa, two-stage, profile likelihood) | new in A2 | `tests/test_interval_audit.py` | Executed A2 (pass). |
| Local-model study | stored aggregates only | no raw data anywhere (see `FINAL_REPRODUCIBILITY_REPORT.md`) | Not reproducible. |

## 2. S0 rerun comparison (A2)

See `FINAL_REPRODUCIBILITY_REPORT.md`, section "S0 rerun in the closure pass", for the exact command, comparison tolerance and result.

## 3. Major claims

| Claim | Manuscript location | Source / script | Type | Fresh | Caveat |
|---|---|---|---|---|---|
| Pooled Wilson covers 68.3% at 10 runs, T=100 (MC s.e. 1.0) | Results, Fig. 1, Tab. 2 | `sim_coverage.py` → `sim_coverage.json` | S | A2 (see §2) | one generator (task sd 1.5, interaction 0.5) |
| Run-level McNemar rejects 17.9% (MC s.e. 0.9) | Results, Fig. 3 | `sim_tests.py` | S | A2 | interaction sd 1, T=100, R=10 |
| Fixed-budget SE ratio and power | Results, Figs 5–6 | `sim_design.py`, `sim_extra.py alloc` | S | A2 (design), stored (alloc) | not a general law; depends on interaction |
| Sequential false positives: peeking 36.5%, CS 1.3% | Results, Fig. 7 | `sim_sequential.py` | S | A1 | 600 reps, MC s.e. ≤ 2 pts |
| Max wrong-decision 2.70%, max false-equivalence 0.55% | Results, App. F Tab. 5 | `audit_equivalence.py` | S | A1 | simulation frequencies, 2000 reps, one three-point family; not guarantees; theory bound α is separate (T, Section 3) |
| ≤ α for any wrong decision, anytime | Section 3 | derivation + `tests/test_sequential_audit.py` | T | tests A1 | i.i.d. bounded tasks; grid guarantee numerical |
| SWE-bench 129 entries, 8 256 pairs, 71.7% Holm, 0/128 adjacent | Results, Tab. 1 | `reanalyze_swebench.py`, `audit_swebench.py` | E | A1 | official convention: unobserved = unresolved; hash of snapshot inferred |
| Missing-log sensitivity 70.8–72.3% | Tab. 7 | `audit_swebench.py` | E | A1 | bound is per pair |
| tau-bench retail 10.2 [3.9,16.5], p 0.002 / Holm 0.005; airline 3.5 | Results, Tab. 6 | `audit_taubench.py` | E | A1 | studentised sign-flip: asymptotic under weak null |
| Majority McNemar p=0.16 not conservative, different estimand | Methods, Results | `audit_taubench.py` | E | A1 | — |
| Local models: 7 models, 8 400 runs, run-noise share 34%, ~387 tasks for a 5-point gap | Results, Fig. 11 | `analyze_real.py` → `real_main.json` | L | stored | raw data unavailable; plug-in estimate, wide uncertainty |
| BCa / two-stage / profile likelihood correct | Methods | `tests/test_interval_audit.py` | T/S | A2 | validity of profile interval needs beta model; coverage held in one two-point-mixture simulation |
