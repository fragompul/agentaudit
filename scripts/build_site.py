"""Assemble the static site: web playground + a zip of the package for Pyodide."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from make_site_data import write as write_site_data

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(ROOT / "web", SITE, ignore=shutil.ignore_patterns("__pycache__"))
    write_site_data(SITE / "data" / "results.json")
    paper = ROOT / "paper" / "agentaudit-paper.pdf"
    if paper.exists():
        (SITE / "paper").mkdir()
        shutil.copy(paper, SITE / "paper" / paper.name)
    (SITE / "py").mkdir()
    pkg = ROOT / "src" / "agentaudit"
    with zipfile.ZipFile(SITE / "py" / "agentaudit.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pkg.rglob("*.py")):
            zf.write(f, f.relative_to(ROOT / "src"))
    print(f"site built at {SITE}")


if __name__ == "__main__":
    main()
