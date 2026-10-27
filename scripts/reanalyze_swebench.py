"""Re-analysis of public SWE-bench Verified entries: how many pairs are distinguishable?

Reads data/raw/swebench_verified/*.json (fetch_swebench.py) and data/raw/swebench_meta/ (fetch_swebench_meta.py);
writes results/reanalysis_swebench.json. Only derived aggregates are stored.

The convention of the official score is resolved/500, so an instance absent from ``resolved`` counts as
unresolved. That is only a faithful reading if the entry was run on all 500 instances. We therefore audit
coverage (public S3 log folders plus ``no_generation``) and repeat the analysis on subsets:
  all            every entry with >= min_resolved resolved instances (the original analysis)
  covered        entries whose log folders plus no_generation cover all 500 instances
  single         entries whose metadata declares attempts == 1 (no best@k)
  covered+single both
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from agentaudit.reanalysis import reanalyze_single_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/swebench_verified")
    ap.add_argument("--meta", default="data/raw/swebench_meta")
    ap.add_argument("--out", default="results/reanalysis_swebench.json")
    ap.add_argument("--min-resolved", type=int, default=50)
    ns = ap.parse_args()
    subs: dict[str, set[str]] = {}
    nogen: dict[str, set[str]] = {}
    universe: set[str] = set()
    for f in sorted(glob.glob(f"{ns.data}/*.json")):
        d = json.loads(Path(f).read_text())
        universe |= set(d.get("resolved", [])) | set(d.get("no_generation", []))
        universe |= set(d.get("no_logs", [])) | set(d.get("generated", []))
        if len(d.get("resolved", [])) >= ns.min_resolved:
            subs[Path(f).stem] = set(d["resolved"])
            nogen[Path(f).stem] = set(d.get("no_generation", []))
    tasks = sorted(universe)
    assert len(tasks) == 500, f"expected the 500 Verified instances, found {len(tasks)}"
    index = {r["name"]: r for r in json.loads(Path(ns.meta, "index.json").read_text())}
    names = sorted(subs)
    covered: dict[str, bool | None] = {}
    attempts: dict[str, str] = {}
    for n in names:
        r = index.get(n, {})
        dirs = set(r.get("s3_instance_dirs", []))
        covered[n] = (len(dirs | nogen[n]) == 500) if (r.get("s3_ok") and dirs) else None
        att = ((r.get("metadata") or {}).get("tags") or {}).get("system", {}) or {}
        attempts[n] = str(att.get("attempts"))
    masks = {
        "all": [True] * len(names),
        "covered": [covered[n] is True for n in names],
        "single": [attempts[n] == "1" for n in names],
        "covered+single": [covered[n] is True and attempts[n] == "1" for n in names],
    }
    out: dict[str, Any] = {
        "source": "github.com/swe-bench/experiments evaluation/verified (accessed via script)",
        "inclusion": f"entries with results/results.json and >= {ns.min_resolved} resolved instances",
        "coverage_audit": {
            "n_entries_included": len(names),
            "coverage_verified": sum(covered[n] is True for n in names),
            "coverage_unverifiable_no_public_logs": sum(covered[n] is None for n in names),
            "coverage_incomplete": sum(covered[n] is False for n in names),
            "attempts_1": sum(attempts[n] == "1" for n in names),
            "attempts_multiple": sum(attempts[n] in {"2+", "2"} for n in names),
            "attempts_unknown": sum(attempts[n] == "None" for n in names),
        },
        "subsets": {},
    }
    for key, mask in masks.items():
        sel = [n for n, m in zip(names, mask, strict=True) if m]
        if len(sel) < 3:
            continue
        x = np.array([[t in subs[n] for t in tasks] for n in sel])
        res = reanalyze_single_run(x, sel)
        out["subsets"][key] = res
        print(
            f"{key:15s} {res['n_agents']:3d} agents {res['n_pairs']:5d} pairs: "
            f"{res['frac_distinguishable_holm']:.1%} dist. (Holm), "
            f"{res['frac_distinguishable_raw']:.1%} raw; adjacent: "
            f"{res['adjacent_pairs']['n_distinguishable_holm_all_pairs']}/{res['adjacent_pairs']['n']} "
            f"(Holm, all pairs), {res['adjacent_pairs']['n_distinguishable_raw']} raw"
        )
    out.update(out["subsets"]["all"])  # backwards compatible top-level = original analysis
    Path(ns.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
