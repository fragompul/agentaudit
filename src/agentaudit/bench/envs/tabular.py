"""Tabular data + calculator environment and task family."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

import numpy as np

from agentaudit.bench.base import (
    Environment,
    Family,
    JSONDict,
    Param,
    Task,
    ToolInvocation,
    ToolSpec,
    handle_common,
    install_common_tools,
)
from agentaudit.bench.words import PRODUCTS, pick

REGIONS = ("north", "south", "east", "west")
_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_FUNCS: dict[str, Callable[..., float]] = {
    "sum": lambda *a: float(sum(a)),
    "min": lambda *a: float(min(a)),
    "max": lambda *a: float(max(a)),
    "round": lambda x, n=0: round(x, int(n)),
    "avg": lambda *a: float(sum(a)) / len(a),
}


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression (numbers, + - * / **, sum/min/max/avg/round)."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("unsupported expression")

    if len(expr) > 400:
        raise ValueError("expression too long")
    return ev(ast.parse(expr.strip(), mode="eval").body)


class TabularEnv(Environment):
    def __init__(self, rows: list[list[Any]]) -> None:
        super().__init__()
        self.rows = rows  # columns: region, product, units, price
        self.register(ToolSpec("list_tables", "List available tables and their columns."))
        self.register(
            ToolSpec(
                "show_rows",
                "Show up to 10 rows of the 'sales' table starting at offset (0-based).",
                {
                    "offset": Param("integer", "Index of the first row"),
                    "limit": Param("integer", "Number of rows (max 10)", required=False),
                },
            )
        )
        self.register(
            ToolSpec(
                "calc",
                "Evaluate an arithmetic expression, e.g. round(sum(1,2,3)/3, 2). "
                "Supports + - * / ** sum min max avg round.",
                {"expression": Param("string", "Expression")},
            )
        )
        install_common_tools(self)

    def _dispatch(self, name: str, args: JSONDict) -> str:
        common = handle_common(self, name, args)
        if common is not None:
            return common
        if name == "list_tables":
            return f"sales (region, product, units, price) - {len(self.rows)} rows"
        if name == "show_rows":
            off, lim = int(args["offset"]), min(int(args.get("limit", 10)), 10)
            if off < 0 or off >= len(self.rows):
                return f"ERROR: offset out of range (table has {len(self.rows)} rows)"
            chunk = self.rows[off : off + lim]
            return "\n".join(
                f"{off + i}: {r[0]}, {r[1]}, units={r[2]}, price={r[3]}"
                for i, r in enumerate(chunk)
            )
        if name == "calc":
            val = safe_eval(str(args["expression"]))
            return str(int(val)) if val == int(val) else f"{val:.6g}"
        raise KeyError(name)

    def state(self) -> JSONDict:
        return {"answer": self.answer}


class TabularFamily(Family):
    name = "tabular"
    TEMPLATES = ("sum_units", "revenue", "max_price", "avg_units", "diff_regions")

    def generate(
        self, rng: np.random.Generator, index: int, horizon: int, distractors: int
    ) -> Task:
        template = self.TEMPLATES[index % len(self.TEMPLATES)]
        n = 6 + 3 * max(horizon, 1) + 3 * distractors
        prods = pick(rng, PRODUCTS, 4)
        rows: list[list[Any]] = [
            [
                str(rng.choice(REGIONS)),
                str(rng.choice(prods)),
                int(rng.integers(1, 30)),
                int(rng.integers(2, 20)),
            ]
            for _ in range(n)
        ]
        region = str(rows[0][0])
        sel = [r for r in rows if r[0] == region]
        if template == "sum_units":
            val = float(sum(r[2] for r in sel))
            q = f"What is the total number of units sold in the {region} region?"
        elif template == "revenue":
            val = float(sum(r[2] * r[3] for r in sel))
            q = (
                f"What is the total revenue (units * price summed over rows) "
                f"in the {region} region?"
            )
        elif template == "max_price":
            val = float(max(r[3] for r in sel))
            q = f"What is the highest unit price among rows of the {region} region?"
        elif template == "avg_units":
            val = round(sum(r[2] for r in sel) / len(sel), 2)
            q = f"What is the average units per row in the {region} region, rounded to 2 decimals?"
        else:
            other = next((str(r[0]) for r in rows if r[0] != region), "north")
            val = float(
                sum(r[2] for r in rows if r[0] == region) - sum(r[2] for r in rows if r[0] == other)
            )
            q = (
                f"How many more units were sold in the {region} region than in the "
                f"{other} region (can be negative)?"
            )
        spec: JSONDict = {"template": template, "rows": rows, "value": val, "region": region}
        pages = (n + 9) // 10
        return Task(
            f"tabular-{index:04d}",
            self.name,
            q + " Use the tools to read the table and calculate; submit the number as the answer.",
            spec,
            pages + 2,
            distractors,
            True,
        )

    def build(self, task: Task) -> TabularEnv:
        return TabularEnv(task.spec["rows"])

    def check(self, task: Task, env: Environment) -> bool:
        assert isinstance(env, TabularEnv)
        if env.answer is None:
            return False
        try:
            got = float(env.answer.replace(",", "").strip())
        except ValueError:
            return False
        return abs(got - float(task.spec["value"])) < 0.011

    def oracle(self, task: Task) -> list[ToolInvocation]:
        return [
            ToolInvocation("show_rows", {"offset": 0}),
            ToolInvocation("calc", {"expression": "1+1"}),
            ToolInvocation("submit_answer", {"answer": str(task.spec["value"])}),
        ]
