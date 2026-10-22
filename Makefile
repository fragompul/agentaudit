PY ?= .venv/bin/python
.PHONY: install lint type test cov reproduce docs clean
install:
	uv venv --python 3.12 .venv && uv pip install -e ".[dev,plots,docs]"
lint:
	.venv/bin/ruff format --check src tests scripts && .venv/bin/ruff check src tests scripts
type:
	.venv/bin/mypy src
test:
	.venv/bin/pytest
cov:
	.venv/bin/pytest --cov=agentaudit --cov-report=term-missing --cov-fail-under=90
docs:
	.venv/bin/mkdocs build --strict
