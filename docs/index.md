# AgentAudit

**Agent leaderboards report one number per agent from one run. AgentAudit tells you how much of that number is signal.**

It is three things:

1. a **library** (`agentaudit`) for statistically rigorous agent evaluation: unbiased pass@k and pass^k,
   intervals that respect the tasks-by-runs structure, variance decomposition, paired tests with
   multiplicity control, ranking uncertainty, a power planner and anytime-valid sequential stopping;
2. a **benchmark** (`AgentAudit-Bench`): 360 procedurally generated tasks in six offline, deterministic
   tool environments, verified by final state, with seeded fault injection;
3. an **empirical study** that quantifies the problem with simulations, local models and re-analysis of public
   results. See [Results](results.md).

Try it in the browser: the [playground](https://fragompul.github.io/agentaudit/) runs this exact library through Pyodide.

```python
from agentaudit.analysis import analyze
from agentaudit.schema import read_jsonl, to_resultset

report = analyze(to_resultset(read_jsonl("trajectories.jsonl")))
```
