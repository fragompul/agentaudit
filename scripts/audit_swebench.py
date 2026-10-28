# ruff: noqa: E501, C408
"""Final audit C: SWE-bench Verified denominators, missing-log handling and sensitivity bounds.

Reads data/raw/swebench_verified (results.json per entry) and data/raw/swebench_meta (index.json).
Convention of the official score: resolved / 500, so every instance not in ``resolved`` counts as
unresolved. ``no_generation`` (the system produced no patch) is a legitimate failure; ``no_logs``
(the harness left no evaluation log) is an *unobserved* outcome that the convention also scores as 0.
We keep the official convention as the headline (it is what a leaderboard reader sees) and quantify
how much it matters:
  official           every entry (>= 50 resolved), unobserved = unresolved (original analysis)
  complete_logs      entries with an empty ``no_logs`` list
  complete+single    complete_logs and declared attempts == 1
  pairwise_complete  per pair, drop the instances in either entry's ``no_logs``
  robust_bound       per pair, the least favourable filling of unobserved cells (leader's unknown
                     stays 0, trailing entry's unknown set to 1); the pair counts as robustly
                     distinguishable only if it is still significant after Holm over all pairs
Writes results/audit_swebench.json.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

from agentaudit.compare import holm
from agentaudit.reanalysis import mcnemar_matrix, reanalyze_single_run


def mc(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    return 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(only_a, only_b), n, 0.5)))


def main() -> None:
    idx = {r["name"]: r for r in json.loads(Path("data/raw/swebench_meta/index.json").read_text())}
    subs, nolog, nogen = {}, {}, {}
    universe: set[str] = set()
    for f in sorted(glob.glob("data/raw/swebench_verified/*.json")):
        d = json.loads(Path(f).read_text())
        universe |= set(d["resolved"]) | set(d.get("no_generation", [])) | set(d.get("no_logs", []))
        universe |= set(d.get("generated", []))
        if len(d["resolved"]) >= 50:
            n = Path(f).stem
            subs[n], nolog[n], nogen[n] = (
                set(d["resolved"]),
                set(d.get("no_logs", [])),
                set(d.get("no_generation", [])),
            )
    tasks = sorted(universe)
    assert len(tasks) == 500
    names = sorted(subs)
    x = np.array([[t in subs[n] for t in tasks] for n in names])
    miss = np.array([[t in nolog[n] for t in tasks] for n in names])
    attempts = {
        n: str(
            (((idx.get(n, {}).get("metadata") or {}).get("tags") or {}).get("system") or {}).get(
                "attempts"
            )
        )
        for n in names
    }
    s3dirs = {n: set(idx.get(n, {}).get("s3_instance_dirs", [])) for n in names}
    s3_cov = {
        n: bool(idx.get(n, {}).get("s3_ok") and s3dirs[n] and len(s3dirs[n] | nogen[n]) == 500)
        for n in names
    }
    out: dict = {
        "n_entries": len(names),
        "n_instances": 500,
        "entries_with_no_logs": int((miss.sum(axis=1) > 0).sum()),
        "no_logs_counts_nonzero": sorted(int(v) for v in miss.sum(axis=1) if v > 0),
        "entries_s3_cov_but_with_no_logs": [n for n in names if s3_cov[n] and nolog[n]],
        "n_s3_cov": sum(s3_cov.values()),
        "subsets": {},
    }
    subsets = {
        "official": np.ones(len(names), bool),
        "complete_logs": miss.sum(axis=1) == 0,
        "complete_logs+single": (miss.sum(axis=1) == 0)
        & np.array([attempts[n] == "1" for n in names]),
        "s3_cov+complete_logs": np.array([s3_cov[n] for n in names]) & (miss.sum(axis=1) == 0),
    }
    for key, m in subsets.items():
        sel = np.flatnonzero(m)
        res = reanalyze_single_run(x[sel], [names[i] for i in sel])
        res.pop("top10")
        out["subsets"][key] = res
        print(
            key,
            res["n_agents"],
            res["n_pairs"],
            round(res["frac_distinguishable_holm"], 4),
            res["adjacent_pairs"],
        )
    # pairwise complete-case and adverse-fill bounds on the full set
    m_ = len(names)
    iu = np.triu_indices(m_, 1)
    p0, _ = mcnemar_matrix(x)
    score = x.mean(axis=1)
    p_cc, p_adv = [], []
    for i, j in zip(*iu, strict=True):
        ok = ~(miss[i] | miss[j])
        a, b = x[i, ok], x[j, ok]
        p_cc.append(mc(int((a & ~b).sum()), int((b & ~a).sum())))
        lead, trail = (i, j) if score[i] >= score[j] else (j, i)
        xl, xt = x[lead].copy(), x[trail].copy()
        xt[miss[trail]] = True  # least favourable for the leader
        xl[miss[lead]] = False
        p_adv.append(mc(int((xl & ~xt).sum()), int((xt & ~xl).sum())))
    p_raw = p0[iu]
    for lab, p in (
        ("official", p_raw),
        ("pairwise_complete", np.array(p_cc)),
        ("robust_bound", np.array(p_adv)),
    ):
        h = holm(p)
        out["subsets"][f"{lab}_all129"] = {
            "n_pairs": int(p.size),
            "n_distinguishable_holm": int((h < 0.05).sum()),
            "frac_distinguishable_holm": float((h < 0.05).mean()),
            "n_distinguishable_raw": int((p < 0.05).sum()),
        }
        print(lab, out["subsets"][f"{lab}_all129"])
    out["pairs_with_unobserved_cells"] = int(
        sum(1 for i, j in zip(*iu, strict=True) if miss[i].any() or miss[j].any())
    )
    # selection effect of the >= 50 resolved inclusion criterion
    allf = [
        json.loads(Path(f).read_text())
        for f in sorted(glob.glob("data/raw/swebench_verified/*.json"))
    ]
    out["inclusion_threshold"] = {
        "files_with_results_json": len(allf),
        "excluded_below_50_resolved": [len(d["resolved"]) for d in allf if len(d["resolved"]) < 50],
    }
    Path("results/audit_swebench.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
