"""Download metadata.yaml and per-instance details of SWE-bench Verified entries and audit log coverage.

For every entry in github.com/swe-bench/experiments/evaluation/verified we fetch ``metadata.yaml`` and,
when present, ``results/results.json`` or ``per_instance_details.json``. The ``assets.logs`` field points
to a public S3 prefix with one folder per evaluated instance; we list it (names only, no downloads of
logs) to count how many of the 500 instances each entry has evaluation logs for. Output: data/raw/swebench_meta/.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

API = "https://api.github.com/repos/swe-bench/experiments/contents/evaluation/verified"
RAW = "https://raw.githubusercontent.com/swe-bench/experiments/main/evaluation/verified/{}/{}"
S3 = "https://swe-bench-submissions.s3.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def get(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "agentaudit-reanalysis"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            data: bytes = r.read()
            return data
    except Exception:
        return None


def s3_instance_dirs(logs_uri: str) -> tuple[list[str], bool]:
    """Instance folder names under an ``s3://swe-bench-submissions/...`` prefix, and ok flag."""
    prefix = logs_uri.removeprefix("s3://swe-bench-submissions/").rstrip("/") + "/"
    names: list[str] = []
    token = ""
    while True:
        q = {"prefix": prefix, "delimiter": "/", "list-type": "2", "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        body = get(S3 + "?" + urllib.parse.urlencode(q))
        if body is None:
            return names, False
        root = ET.fromstring(body)
        for cp in root.iter(NS + "CommonPrefixes"):
            p = cp.find(NS + "Prefix")
            if p is not None and p.text:
                names.append(p.text[len(prefix) :].strip("/"))
        nxt = root.find(NS + "NextContinuationToken")
        if nxt is None or not nxt.text:
            return names, True
        token = nxt.text


def one(name: str, out: Path) -> dict[str, object]:
    meta_raw = get(RAW.format(name, "metadata.yaml"))
    rec: dict[str, object] = {"name": name, "metadata": None}
    if meta_raw is not None:
        meta = yaml.safe_load(meta_raw)
        rec["metadata"] = meta
        (out / f"{name}.metadata.yaml").write_bytes(meta_raw)
        logs = (meta.get("assets") or {}).get("logs")
        if isinstance(logs, str) and logs.startswith("s3://swe-bench-submissions/"):
            dirs, ok = s3_instance_dirs(logs)
            rec["s3_logs_prefix"] = logs
            rec["s3_instance_dirs"] = sorted(dirs)
            rec["s3_ok"] = ok
    for rel in ("results/results.json", "per_instance_details.json"):
        body = get(RAW.format(name, rel))
        if body is not None:
            (out / f"{name}.{Path(rel).name}").write_bytes(body)
            rec["has_" + Path(rel).stem] = True
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/swebench_meta")
    ns = ap.parse_args()
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    listing = get(API)
    assert listing is not None
    names = [e["name"] for e in json.loads(listing) if e["type"] == "dir"]
    with ThreadPoolExecutor(12) as ex:
        recs = list(ex.map(lambda n: one(n, out), names))
    (out / "index.json").write_text(json.dumps(recs, indent=1, default=str))
    ok = sum(1 for r in recs if r.get("s3_ok"))
    print(f"{len(recs)} entries; S3 logs listed for {ok}")


if __name__ == "__main__":
    main()
