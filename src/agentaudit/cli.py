"""Command line interface: ``agentaudit <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from agentaudit import __version__
from agentaudit.analysis import analyze
from agentaudit.report import to_html, to_markdown
from agentaudit.results import ResultSet
from agentaudit.schema import read_jsonl, to_resultset


def _load(path: str) -> ResultSet:
    p = Path(path)
    if p.suffix == ".json":
        return ResultSet.load(p)
    return to_resultset(read_jsonl(p))


def cmd_analyze(ns: argparse.Namespace) -> int:
    rs = _load(ns.input)
    if ns.models:
        rs = rs.select(ns.models)
    a = analyze(rs, level=ns.level, n_boot=ns.n_boot, seed=ns.seed, mde_target=ns.delta)
    if ns.json:
        Path(ns.json).write_text(json.dumps(a, indent=1))
    if ns.html:
        Path(ns.html).write_text(to_html(a, ns.title))
    md = to_markdown(a, ns.title)
    if ns.markdown:
        Path(ns.markdown).write_text(md)
    if not (ns.json or ns.html or ns.markdown) or ns.print:
        sys.stdout.write(md)
    return 0


def cmd_power(ns: argparse.Namespace) -> int:
    from agentaudit.power import DiffVariance, mde, optimal_runs, plan

    if ns.input:
        rs = _load(ns.input)
        if rs.n_models < 2:
            sys.stderr.write("need at least two agents in the pilot\n")
            return 2
        dv = DiffVariance.from_pilot(rs.scores[0], rs.scores[1])
    else:
        dv = DiffVariance(ns.task_var, ns.run_var)
    rows = plan(ns.delta, dv, cost_task=ns.cost_task, cost_run=ns.cost_run, target_power=ns.power)
    sys.stdout.write(
        f"diff variance: task {dv.task:.4f}, run {dv.run:.4f}; target delta {ns.delta:.3f}, "
        f"power {ns.power:.0%}\n"
    )
    sys.stdout.write("runs/task  tasks  attempts      cost\n")
    for r in rows:
        sys.stdout.write(f"{r.n_runs:8d} {r.n_tasks:6d} {r.total_attempts:9d} {r.cost:9.1f}\n")
    sys.stdout.write(
        f"cost-optimal runs per task: {optimal_runs(dv, ns.cost_task, ns.cost_run)}; "
        f"MDE with 100 tasks x 5 runs: {mde(100, 5, dv):.3f}\n"
    )
    return 0


def cmd_bench(ns: argparse.Namespace) -> int:
    from agentaudit.bench.suite import audit_suite, core_suite, generate_suite, save_suite

    tasks = generate_suite(ns.per_family, ns.seed)
    if ns.action == "audit":
        bad = 0
        for fam, rep in audit_suite(tasks).items():
            status = "ok" if rep.ok else "FAILED"
            sys.stdout.write(f"{fam:8s} {rep.n_tasks:4d} tasks  {status}\n")
            bad += not rep.ok
        return 1 if bad else 0
    if ns.core:
        tasks = core_suite(tasks)
    save_suite(tasks, ns.out)
    sys.stdout.write(f"wrote {len(tasks)} tasks to {ns.out}\n")
    return 0


def cmd_run(ns: argparse.Namespace) -> int:
    from agentaudit.agents.cache import CachedAgent, Store
    from agentaudit.agents.synthetic import SkillAgent
    from agentaudit.agents.text_protocol import TextProtocolAgent, ollama_chat
    from agentaudit.bench.suite import core_suite, generate_suite
    from agentaudit.runner import RunConfig, run_evaluation

    tasks = generate_suite(60, ns.seed)
    if ns.suite == "core":
        tasks = core_suite(tasks)
    agent: SkillAgent | TextProtocolAgent | CachedAgent
    if ns.synthetic is not None:
        agent = SkillAgent(f"synthetic-{ns.synthetic:+.1f}", ns.synthetic)
    else:
        agent = TextProtocolAgent(
            ns.model, ollama_chat(ns.model, temperature=ns.temperature), {"model": ns.model}
        )
        if ns.cache:
            agent = CachedAgent(agent, Store(ns.cache))
    cfg = RunConfig(fault_rate=ns.fault_rate, base_seed=ns.seed)
    n = run_evaluation(agent, tasks, ns.runs, ns.out, cfg, workers=ns.workers)
    sys.stdout.write(f"{n} new attempts appended to {ns.out}\n")
    return 0


def cmd_failures(ns: argparse.Namespace) -> int:
    from agentaudit.bench.suite import generate_suite
    from agentaudit.failures import CATEGORIES, failure_table

    answer_tasks = {t.task_id for t in generate_suite(60, 0) if t.requires_answer}
    tab = failure_table(read_jsonl(ns.input), answer_tasks)
    sys.stdout.write("model".ljust(22) + " ".join(c[:14].rjust(15) for c in CATEGORIES) + "\n")
    for model, counts in sorted(tab.items()):
        sys.stdout.write(
            model.ljust(22) + " ".join(str(counts[c]).rjust(15) for c in CATEGORIES) + "\n"
        )
    return 0


def cmd_demo(ns: argparse.Namespace) -> int:
    from agentaudit.simulate import SyntheticPopulation

    pop = SyntheticPopulation(
        (0.0, 0.15, 0.3, 1.0), names=("agent-A", "agent-B", "agent-C", "agent-D")
    )
    rs = pop.sample(ns.tasks, ns.runs, np.random.default_rng(ns.seed))
    sys.stdout.write(to_markdown(analyze(rs, n_boot=1000), "AgentAudit demo (synthetic agents)"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="agentaudit", description="Statistically rigorous agent evaluation"
    )
    ap.add_argument("--version", action="version", version=f"agentaudit {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="audit a trajectories JSONL (or ResultSet .json)")
    a.add_argument("input")
    a.add_argument("--models", nargs="*")
    a.add_argument("--level", type=float, default=0.95)
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--delta", type=float, default=0.05, help="target difference for power planning")
    a.add_argument("--title", default="AgentAudit report")
    a.add_argument("--json")
    a.add_argument("--html")
    a.add_argument("--markdown")
    a.add_argument("--print", action="store_true", help="also print Markdown to stdout")
    a.set_defaults(func=cmd_analyze)

    p = sub.add_parser("power", help="how many tasks and runs are needed?")
    p.add_argument("--input", help="pilot trajectories (first two agents are compared)")
    p.add_argument("--task-var", type=float, default=0.10)
    p.add_argument("--run-var", type=float, default=0.15)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--cost-task", type=float, default=0.0)
    p.add_argument("--cost-run", type=float, default=1.0)
    p.set_defaults(func=cmd_power)

    b = sub.add_parser("bench", help="generate or audit AgentAudit-Bench")
    b.add_argument("action", choices=["generate", "audit"])
    b.add_argument("--per-family", type=int, default=60)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--core", action="store_true")
    b.add_argument("--out", default="suite.jsonl")
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser("run", help="run an agent on the benchmark (resumable)")
    r.add_argument("--model", default="qwen2.5:3b", help="Ollama model tag")
    r.add_argument("--synthetic", type=float, help="use a synthetic agent of this latent skill")
    r.add_argument("--suite", choices=["core", "full"], default="core")
    r.add_argument("--runs", type=int, default=5)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--fault-rate", type=float, default=0.0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--cache", help="record/replay cache file")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    f = sub.add_parser("failures", help="failure taxonomy of a trajectories file")
    f.add_argument("input")
    f.set_defaults(func=cmd_failures)

    d = sub.add_parser("demo", help="analyse simulated agents (no model needed)")
    d.add_argument("--tasks", type=int, default=100)
    d.add_argument("--runs", type=int, default=5)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_demo)
    return ap


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    code: int = ns.func(ns)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
