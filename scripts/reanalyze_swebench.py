"""Re-analysis of public SWE-bench Verified submissions: how many pairs are distinguishable?

Reads data/raw/swebench_verified/*.json (see fetch_swebench.py); writes
results/reanalysis_swebench.json. Only derived aggregates are stored.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from agentaudit.reanalysis import reanalyze_single_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/swebench_verified")
    ap.add_argument("--out", default="results/reanalysis_swebench.json")
    ap.add_argument("--min-resolved", type=int, default=50)
    ns = ap.parse_args()
    subs: dict[str, set[str]] = {}
    universe: set[str] = set()
    for f in sorted(glob.glob(f"{ns.data}/*.json")):
        d = json.loads(Path(f).read_text())
        universe |= (
            set(d.get("generated", []))
            | set(d.get("no_generation", []))
            | set(d.get("resolved", []))
        )
        if len(d.get("resolved", [])) >= ns.min_resolved:
            subs[Path(f).stem] = set(d["resolved"])
    tasks = sorted(universe)
    assert len(tasks) == 500, f"expected the 500 Verified instances, found {len(tasks)}"
    names = sorted(subs)
    x = np.array([[t in subs[n] for t in tasks] for n in names])
    out = reanalyze_single_run(x, names)
    out["source"] = "github.com/swe-bench/experiments evaluation/verified (accessed via script)"
    out["inclusion"] = f"submissions with >= {ns.min_resolved} resolved instances"
    Path(ns.out).write_text(json.dumps(out, indent=1))
    print(
        f"{out['n_agents']} agents, {out['n_pairs']} pairs: "
        f"{out['frac_distinguishable_holm']:.1%} distinguishable (Holm), "
        f"{out['frac_distinguishable_raw']:.1%} unadjusted; adjacent leaderboard pairs: "
        f"{out['adjacent_pairs']['frac_distinguishable_holm_within_adjacent']:.1%}"
    )


if __name__ == "__main__":
    main()
