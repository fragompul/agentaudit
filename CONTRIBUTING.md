# Contributing

```bash
make install        # venv + editable install with dev, plots and docs extras
make lint type test # ruff, mypy --strict, pytest (coverage gate: 90%)
```

* One logical change per commit, Conventional Commits (`feat(metrics): ...`, `fix:`, `test:`, `docs:`, `exp:`, `paper:`).
* Every statistical function needs (i) a numerical check against SciPy/statsmodels where one exists and
  (ii) a property-based or simulation test of its defining property (coverage, unbiasedness, monotonicity).
* New environments must pass `agentaudit.bench.audit`: the oracle solves every generated task, a do-nothing
  agent solves none, and generation is deterministic in the seed. See `docs/extending.md`.
* The test suite must run without network access.
