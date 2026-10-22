# Architecture

```
schema ──► results (ResultSet: models x tasks x runs)
                │
 metrics ◄──────┤────► intervals, variance, compare, ranking, power, sequential
                │                       │
             analysis  ──► report (md/html) ──► cli, web (Pyodide)

bench (envs, tasks, fault injection) ─► runner ◄─ agents (text protocol, HTTP, synthetic, cache)
                                           │
                                  trajectories JSONL ─► failures
ingest (tau-bench, SWE-bench) ──────────────► results
```

The statistical core depends only on NumPy and SciPy; everything model-related uses the standard library.
