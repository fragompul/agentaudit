"""Glue between the browser UI and the agentaudit library (runs inside Pyodide)."""

import json

from agentaudit.analysis import analyze
from agentaudit.power import DiffVariance, mde, optimal_runs, plan
from agentaudit.report import to_html
from agentaudit.schema import SchemaError, Trajectory, to_resultset

LAST = {}


def run_analysis(text, level=0.95, n_boot=1500, title="AgentAudit report"):
    """Parse trajectories JSONL text, analyse, return JSON {html, summary} or {error}."""
    try:
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(Trajectory.from_dict(json.loads(line)))
                except (ValueError, KeyError, TypeError) as exc:
                    raise SchemaError(f"line {i}: {exc}") from exc
        rs = to_resultset(rows)
        if rs.n_models < 1 or rs.n_tasks < 2:
            raise SchemaError("need at least 1 agent and 2 tasks")
        a = analyze(rs, level=float(level), n_boot=int(n_boot))
        LAST["analysis"] = a
        dv = a.get("power", {}).get("diff_variance")
        return json.dumps(
            {
                "html": to_html(a, title),
                "diff_variance": dv,
                "n_runs": a["n_runs"],
                "n_tasks": a["n_tasks"],
                "n_models": a["n_models"],
            }
        )
    except Exception as exc:  # surface any problem to the UI instead of a Python traceback
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def run_power(delta, task_var, run_var, cost_task, cost_run, power):
    dv = DiffVariance(float(task_var), float(run_var))
    rows = plan(
        float(delta),
        dv,
        cost_task=float(cost_task),
        cost_run=float(cost_run),
        target_power=float(power),
        runs_grid=(1, 2, 3, 5, 10, 20),
    )
    return json.dumps(
        {
            "rows": [r.__dict__ for r in rows],
            "optimal_runs": optimal_runs(dv, float(cost_task), float(cost_run)),
            "mde_100x5": mde(100, 5, dv),
        }
    )
