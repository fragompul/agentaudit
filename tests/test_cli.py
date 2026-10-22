import json
from pathlib import Path

import pytest

from agentaudit.cli import main


def run(*args: str) -> int:
    return main(list(args))


def test_demo_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("demo", "--tasks", "30", "--runs", "3") == 0
    assert "Ranking uncertainty" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run("--version")


def test_bench_generate_audit_and_synthetic_run_analyze(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("bench", "audit", "--per-family", "10") == 0
    assert "ok" in capsys.readouterr().out
    suite = tmp_path / "s.jsonl"
    assert run("bench", "generate", "--core", "--per-family", "9", "--out", str(suite)) == 0
    assert len(suite.read_text().splitlines()) == 18
    out = tmp_path / "t.jsonl"
    for skill in ("0.0", "1.0"):
        assert run("run", "--synthetic", skill, "--runs", "3", "--out", str(out)) == 0
    html, md, js = tmp_path / "r.html", tmp_path / "r.md", tmp_path / "r.json"
    code = run(
        "analyze", str(out), "--n-boot", "200", "--html", str(html), "--markdown", str(md),
        "--json", str(js), "--print",
    )  # fmt: skip
    assert code == 0 and html.read_text().startswith("<!doctype html>")
    assert "synthetic" in md.read_text() and json.loads(js.read_text())["n_models"] == 2
    assert "Scores with" in capsys.readouterr().out
    assert run("failures", str(out)) == 0
    assert run("power", "--input", str(out), "--delta", "0.1") == 0
    text = capsys.readouterr().out
    assert "cost-optimal runs" in text


def test_power_without_pilot_and_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("power", "--task-var", "0.05", "--run-var", "0.2", "--cost-task", "10") == 0
    assert "runs/task" in capsys.readouterr().out
    one = tmp_path / "one.jsonl"
    run("run", "--synthetic", "0.0", "--runs", "2", "--out", str(one))
    assert run("power", "--input", str(one)) == 2
    assert run("analyze", str(one), "--models", "synthetic-+0.0", "--print") == 0
