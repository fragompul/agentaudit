"""Evaluate local Ollama models on AgentAudit-Bench with resumable, balanced, cached runs.

Example (the main experiment):
    python scripts/run_real.py --models llama3.2-1b qwen2.5-3b --runs 10 --workers 4

Re-running the same command resumes where it stopped. All model turns are recorded in a
record/replay cache so that results can be regenerated without any model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from agentaudit.agents.cache import CachedAgent, Store
from agentaudit.agents.text_protocol import TextProtocolAgent, ollama_chat
from agentaudit.bench.suite import core_suite, generate_suite
from agentaudit.runner import RunConfig, run_evaluation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--config", default="configs/models.json")
    ap.add_argument("--suite", choices=["core", "full"], default="core")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--fault-rate", type=float, default=0.0)
    ap.add_argument("--max-tasks", type=int, default=0, help="debug: first N tasks only")
    ap.add_argument("--condition", default="main", help="subdirectory under --out")
    ap.add_argument("--out", default="runs/real")
    ap.add_argument("--seed", type=int, default=0)
    ns = ap.parse_args()

    specs = json.loads(Path(ns.config).read_text())
    tasks = generate_suite(60, seed=ns.seed)
    if ns.suite == "core":
        tasks = core_suite(tasks)
    if ns.max_tasks:
        tasks = tasks[: ns.max_tasks]
    cfg = RunConfig(fault_rate=ns.fault_rate, base_seed=ns.seed)
    out_dir = Path(ns.out) / ns.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "condition.json").write_text(
        json.dumps({**vars(ns), "n_tasks": len(tasks)}, indent=1)
    )
    for alias in ns.models:
        spec = specs[alias]
        chat = ollama_chat(spec["ollama"], temperature=ns.temperature)
        inner = TextProtocolAgent(
            alias,
            chat,
            {
                "model": spec["ollama"],
                "temperature": ns.temperature,
                "num_predict": 256,
                "stop": "tool_call",
            },
        )
        agent = CachedAgent(inner, Store(out_dir / "cache" / f"{alias}.jsonl"))
        t0 = time.time()

        def progress(done: int, total: int, alias: str = alias, t0: float = t0) -> None:
            if done % 25 == 0 or done == total:
                el = time.time() - t0
                eta = (total - done) * el / done / 60
                print(f"[{alias}] {done}/{total}  {el / 60:.1f} min  eta {eta:.1f} min", flush=True)

        n = run_evaluation(
            agent,
            tasks,
            ns.runs,
            out_dir / f"{alias}.jsonl",
            cfg,
            workers=ns.workers,
            progress=progress,
        )
        print(
            f"[{alias}] finished {n} new attempts in {(time.time() - t0) / 60:.1f} min", flush=True
        )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
