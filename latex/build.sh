#!/bin/sh
# Regenerate assets from results/ and compile the paper.
set -e
cd "$(dirname "$0")/.."
python scripts/make_paper_assets.py
cd latex
tectonic main.tex
