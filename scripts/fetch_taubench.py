"""Download tau-bench historical trajectories (MIT licensed; sierra-research/tau-bench)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE = (
    "https://raw.githubusercontent.com/sierra-research/tau-bench/main/"
    "historical_trajectories/{}.json"
)
FILES = ["gpt-4o-airline", "gpt-4o-retail", "sonnet-35-new-airline", "sonnet-35-new-retail"]


def main(out: str = "data/raw/taubench") -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = Path(out) / f"{name}.json"
        if not dest.exists():
            with urllib.request.urlopen(BASE.format(name), timeout=120) as r:  # noqa: S310
                dest.write_bytes(r.read())
        print(dest, dest.stat().st_size)


if __name__ == "__main__":
    main()
