"""Write the example trajectory files bundled with the web demo (small, derived, outcome-only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agentaudit.ingest import taubench_resultset
from agentaudit.schema import from_resultset, write_jsonl
from agentaudit.simulate import SyntheticPopulation

OUT = Path("web/examples")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for domain in ("airline", "retail"):
        src = Path("data/raw/taubench")
        rs = taubench_resultset(
            {
                "gpt-4o": src / f"gpt-4o-{domain}.json",
                "claude-3.5-sonnet": src / f"sonnet-35-new-{domain}.json",
            }
        )
        write_jsonl(
            from_resultset(rs, version=f"tau-bench-{domain}"), OUT / f"taubench_{domain}.jsonl"
        )
    pop = SyntheticPopulation(
        (0.0, 0.15, 0.3, 1.0), names=("agent-A", "agent-B", "agent-C", "agent-D")
    )
    rs = pop.sample(100, 5, np.random.default_rng(7))
    write_jsonl(from_resultset(rs, version="synthetic"), OUT / "synthetic.jsonl")
    for p in sorted(OUT.glob("*.jsonl")):
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
