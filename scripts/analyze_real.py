"""Analyse a directory of trajectory files (one JSONL per model) from scripts/run_real.py.

Writes results/real_<condition>.json: statistical audit, per-family scores, failure taxonomy and the
tokens / latency profile of every model (cost is measured in tokens and time, never money).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from agentaudit.analysis import analyze
from agentaudit.bench.suite import generate_suite
from agentaudit.failures import CATEGORIES, failure_table
from agentaudit.intervals import bootstrap_ci
from agentaudit.metrics import batch_success
from agentaudit.results import ResultSet
from agentaudit.schema import Trajectory, read_jsonl, to_resultset


def load(directory: Path, min_runs: int) -> tuple[list[Trajectory], ResultSet]:
    trajs: list[Trajectory] = []
    for f in sorted(directory.glob("*.jsonl")):
        trajs += list(read_jsonl(f))
    rs = to_resultset(trajs, drop_incomplete_tasks=True)
    complete = [
        m for i, m in enumerate(rs.models) if int(np.sum(~np.isnan(rs.scores[i, 0]))) >= min_runs
    ]
    return trajs, rs.select(complete) if complete else rs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/real/main")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-runs", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=4000)
    ns = ap.parse_args()
    d = Path(ns.dir)
    trajs, rs = load(d, ns.min_runs)
    suite = {t.task_id: t for t in generate_suite(60, 0)}
    keep = set(rs.models)
    trajs = [t for t in trajs if t.model in keep and t.task_id in set(rs.tasks)]
    a = analyze(rs, n_boot=ns.n_boot)

    fam: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in trajs:
        fam[t.model][suite[t.task_id].family].append(t.success)
    by_family = {m: {f: float(np.mean(v)) for f, v in fs.items()} for m, fs in fam.items()}

    prof: dict[str, dict[str, float]] = {}
    for m in rs.models:
        ts = [t for t in trajs if t.model == m]
        prof[m] = {
            "mean_tokens": float(np.mean([t.total_tokens for t in ts])),
            "mean_latency_s": float(np.mean([t.latency_s for t in ts])),
            "mean_steps": float(np.mean([len(t.steps) for t in ts])),
            "frac_step_limit": float(np.mean([t.termination == "step_limit" for t in ts])),
            "frac_error": float(np.mean([t.termination.startswith("error") for t in ts])),
        }
        # per-task mean tokens, for a task-bootstrap interval on the cost axis
        per_task = defaultdict(list)
        for t in ts:
            per_task[t.task_id].append(t.total_tokens)
        tok = np.array([[np.mean(per_task[tid])] for tid in rs.tasks], dtype=float)
        iv = bootstrap_ci(tok / max(tok.max(), 1), batch_success, n_boot=1000)
        prof[m]["tokens_ci"] = [iv.lo * tok.max(), iv.hi * tok.max()]
        me = a["models"][list(rs.models).index(m)]
        prof[m]["score_ci"] = [me["lo"], me["hi"]]

    answer_tasks = {tid for tid, t in suite.items() if t.requires_answer}
    ft = failure_table(trajs, answer_tasks)
    out = {
        "dir": str(d),
        "models": list(rs.models),
        "n_tasks": rs.n_tasks,
        "n_runs": rs.n_runs,
        "analysis": a,
        "by_family": by_family,
        "profile": prof,
        "failure_categories": list(CATEGORIES),
        "failures": {m: dict(c) for m, c in ft.items()},
        "n_trajectories": len(trajs),
    }
    dest = Path(ns.out or f"results/real_{d.name}.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}: {len(rs.models)} models, {rs.n_tasks} tasks, {rs.n_runs} runs")
    for m in a["models"]:
        print(
            f"  {m['model']:18s} {100 * m['score']:5.1f}% [{100 * m['lo']:.1f}, {100 * m['hi']:.1f}]"
        )


if __name__ == "__main__":
    main()
