"""Download per-instance results of public SWE-bench *Verified* submissions (no redistribution).

Source: github.com/swe-bench/experiments, evaluation/verified/<submission>/results/results.json.
The repository declares no license file; we download the small per-submission results files for
analysis only, keep them out of git (data/raw is ignored) and publish only derived aggregates.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.github.com/repos/swe-bench/experiments/contents/evaluation/verified"
RAW = (
    "https://raw.githubusercontent.com/swe-bench/experiments/main/"
    "evaluation/verified/{}/results/results.json"
)


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "agentaudit-reanalysis"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        data: bytes = r.read()
        return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/swebench_verified")
    ns = ap.parse_args()
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [e["name"] for e in json.loads(get(API)) if e["type"] == "dir"]

    def fetch(name: str) -> tuple[str, bool]:
        dest = out / f"{name}.json"
        if dest.exists():
            return name, True
        try:
            dest.write_bytes(get(RAW.format(name)))
            return name, True
        except Exception:
            return name, False

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(fetch, names))
    ok = sum(1 for _, k in res if k)
    print(f"downloaded {ok}/{len(names)} submissions to {out}")


if __name__ == "__main__":
    main()
