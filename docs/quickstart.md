# Quickstart

```bash
pip install agentaudit            # numpy + scipy only
agentaudit demo                   # analyse four simulated agents
```

## Audit your own results

Write one JSON object per attempt (`model`, `task_id`, `run`, `seed`, `success` in [0, 1]) and run

```bash
agentaudit analyze trajectories.jsonl --html report.html --json report.json
```

The report contains task-bootstrap intervals, pass@k vs pass^k, rank probabilities, the share of variance due to
agent, task, agent-by-task and run noise, the fraction of agent pairs that cannot be told apart, and the
minimum detectable difference of your design.

## Plan an evaluation

```bash
agentaudit power --task-var 0.10 --run-var 0.15 --delta 0.05 --cost-task 10
```

## Run the benchmark with a local model

```bash
ollama pull qwen2.5:3b
agentaudit run --model qwen2.5:3b --runs 5 --workers 2 --cache cache.jsonl --out qwen3b.jsonl
agentaudit failures qwen3b.jsonl
```

`run` is resumable (re-run the same command to continue) and `--cache` records every model turn so the
experiment can be replayed without the model.

## Python API

```python
from agentaudit import metrics, intervals, compare, ranking, power, sequential
```
