"""SQLite environment (in-memory, deterministic) and task family."""

from __future__ import annotations

import sqlite3
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
from agentaudit.bench.words import CITIES, PRODUCTS, TOPICS, pick, unique_people

SCHEMA = {
    "employees": "id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary INTEGER, city TEXT",
    "orders": "id INTEGER PRIMARY KEY, customer TEXT, product TEXT, qty INTEGER, price INTEGER",
}


class SqlEnv(Environment):
    def __init__(self, tables: dict[str, list[list[Any]]], decoys: int = 0) -> None:
        super().__init__()
        self.db = sqlite3.connect(":memory:")
        self.schemas = dict(SCHEMA)
        for i in range(decoys):  # distractor tables with confusable names
            self.schemas[f"employees_archive_{i}"] = SCHEMA["employees"]
        for name, ddl in self.schemas.items():
            self.db.execute(f"CREATE TABLE {name} ({ddl})")
        for name, rows in tables.items():
            if rows:
                marks = ",".join("?" * len(rows[0]))
                self.db.executemany(f"INSERT INTO {name} VALUES ({marks})", rows)
        self.register(ToolSpec("list_tables", "List all tables in the database."))
        self.register(
            ToolSpec(
                "describe",
                "Show the columns of a table.",
                {"table": Param("string", "Table name")},
            )
        )
        sql = Param("string", "A single SQL statement")
        self.register(
            ToolSpec("query", "Run a read-only SELECT statement (max 25 rows).", {"sql": sql})
        )
        self.register(
            ToolSpec("execute", "Run a single INSERT, UPDATE or DELETE statement.", {"sql": sql})
        )
        install_common_tools(self)

    def _dispatch(self, name: str, args: JSONDict) -> str:
        common = handle_common(self, name, args)
        if common is not None:
            return common
        if name == "list_tables":
            return "\n".join(self.schemas)
        if name == "describe":
            t = str(args["table"])
            if t not in self.schemas:
                return f"ERROR: no such table: {t}"
            return self.schemas[t].replace(", ", "\n")
        stmt = str(args["sql"]).strip().rstrip(";")
        low = stmt.lower()
        try:
            if name == "query":
                if not low.startswith("select"):
                    return "ERROR: only SELECT statements are allowed here; use execute for changes"
                rows = self.db.execute(stmt).fetchmany(25)
                return "\n".join(" | ".join(map(str, r)) for r in rows) or "(no rows)"
            if not low.startswith(("insert", "update", "delete")):
                return "ERROR: execute only accepts INSERT, UPDATE or DELETE"
            cur = self.db.execute(stmt)
            self.db.commit()
            return f"OK, {cur.rowcount} row(s) affected"
        except sqlite3.Error as exc:
            return f"ERROR: SQL error: {exc}"

    def dump(self, table: str) -> list[list[Any]]:
        return [list(r) for r in self.db.execute(f"SELECT * FROM {table} ORDER BY id")]

    def state(self) -> JSONDict:
        return {t: self.dump(t) for t in self.schemas} | {"answer": self.answer}


class SqlFamily(Family):
    name = "sql"
    TEMPLATES = ("count", "revenue", "raise", "delete_orders", "insert")

    def _tables(self, rng: np.random.Generator, n_rows: int) -> dict[str, list[list[Any]]]:
        people = unique_people(rng, n_rows)
        depts = pick(rng, TOPICS, 4)
        emp = [
            [
                i + 1,
                people[i],
                str(rng.choice(depts)),
                int(rng.integers(20, 90)) * 1000,
                str(rng.choice(CITIES[:6])),
            ]
            for i in range(n_rows)
        ]
        cust = unique_people(rng, 6)
        orders = [
            [
                i + 1,
                str(rng.choice(cust)),
                str(rng.choice(PRODUCTS[:5])),
                int(rng.integers(1, 9)),
                int(rng.integers(2, 40)),
            ]
            for i in range(n_rows)
        ]
        return {"employees": emp, "orders": orders}

    def generate(
        self, rng: np.random.Generator, index: int, horizon: int, distractors: int
    ) -> Task:
        template = self.TEMPLATES[index % len(self.TEMPLATES)]
        tables = self._tables(rng, 12 + 4 * distractors)
        emp, orders = tables["employees"], tables["orders"]
        answer: str | None = None
        expect: dict[str, list[list[Any]]] = {}
        if template == "count":
            dept = str(emp[0][2])
            sal = sorted(r[3] for r in emp if r[2] == dept)
            thr = (sal[int(rng.integers(0, len(sal)))] - 1000) // 1000 * 1000  # >= 1 match
            answer = str(sum(1 for r in emp if r[2] == dept and r[3] > thr))
            instruction = (
                f"How many employees in the '{dept}' department earn more than {thr}? "
                "Submit the number as the answer."
            )
        elif template == "revenue":
            prod = str(orders[0][2])
            answer = str(sum(r[3] * r[4] for r in orders if r[2] == prod))
            instruction = (
                f"What is the total revenue (sum of qty * price) of all orders for "
                f"product '{prod}'? Submit the number as the answer."
            )
        elif template == "raise":
            dept, inc = str(emp[1][2]), int(rng.integers(1, 9)) * 500
            expect["employees"] = [
                [*r[:3], r[3] + inc if r[2] == dept else r[3], r[4]] for r in emp
            ]
            instruction = (
                f"Give every employee in the '{dept}' department a raise of {inc} "
                "(add it to the salary)."
            )
        elif template == "delete_orders":
            cust = str(orders[0][1])
            q = min(r[3] for r in orders if r[1] == cust) + 1 + int(rng.integers(0, 3))
            expect["orders"] = [r for r in orders if not (r[1] == cust and r[3] < q)]
            instruction = f"Delete all orders by customer '{cust}' with quantity below {q}."
        else:  # insert
            k = max(1, horizon)
            new_people = unique_people(rng, 40)
            new_people = [p for p in new_people if p not in {r[1] for r in emp}][:k]
            rows = [
                [len(emp) + 1 + i, p, str(emp[0][2]), 41000 + 1000 * i, str(CITIES[i % 3])]
                for i, p in enumerate(new_people)
            ]
            expect["employees"] = [*emp, *rows]
            listing = "; ".join(
                f"id {r[0]}: {r[1]}, dept {r[2]}, salary {r[3]}, city {r[4]}" for r in rows
            )
            instruction = f"Insert these new rows into the employees table: {listing}."
        spec: JSONDict = {
            "template": template,
            "tables": tables,
            "decoys": distractors,
            "expect": expect,
            "answer": answer,
        }
        h = {"count": 3, "revenue": 3, "raise": 2, "delete_orders": 2, "insert": max(1, horizon)}[
            template
        ]
        return Task(
            f"sql-{index:04d}", self.name, instruction, spec, h, distractors, answer is not None
        )

    def build(self, task: Task) -> SqlEnv:
        return SqlEnv(task.spec["tables"], task.spec["decoys"])

    def check(self, task: Task, env: Environment) -> bool:
        assert isinstance(env, SqlEnv)
        spec = task.spec
        if spec["answer"] is not None:
            return env.answer is not None and env.answer.replace(",", "").strip() == spec["answer"]
        for table, rows in spec["tables"].items():
            want = spec["expect"].get(table, rows)
            if env.dump(table) != [list(r) for r in want]:
                return False
        return True

    def oracle(self, task: Task) -> list[ToolInvocation]:
        spec = task.spec
        t = spec["template"]
        if t == "count":
            dept, thr = (
                _quoted(task.instruction)[0],
                task.instruction.split("more than ")[1].split("?")[0],
            )
            sql = f"SELECT COUNT(*) FROM employees WHERE dept='{dept}' AND salary>{thr}"
            return [
                ToolInvocation("query", {"sql": sql}),
                ToolInvocation("submit_answer", {"answer": spec["answer"]}),
            ]
        if t == "revenue":
            prod = _quoted(task.instruction)[0]
            sql = f"SELECT SUM(qty*price) FROM orders WHERE product='{prod}'"
            return [
                ToolInvocation("query", {"sql": sql}),
                ToolInvocation("submit_answer", {"answer": spec["answer"]}),
            ]
        if t == "raise":
            dept = _quoted(task.instruction)[0]
            inc = task.instruction.split("raise of ")[1].split(" ")[0]
            sql = f"UPDATE employees SET salary=salary+{inc} WHERE dept='{dept}'"
        elif t == "delete_orders":
            cust = _quoted(task.instruction)[0]
            q = task.instruction.split("below ")[1].rstrip(".")
            sql = f"DELETE FROM orders WHERE customer='{cust}' AND qty<{q}"
        else:
            new = spec["expect"]["employees"][len(spec["tables"]["employees"]) :]
            return [
                ToolInvocation(
                    "execute",
                    {
                        "sql": "INSERT INTO employees VALUES "
                        f"({r[0]}, '{r[1]}', '{r[2]}', {r[3]}, '{r[4]}')"
                    },
                )
                for r in new
            ]
        return [ToolInvocation("execute", {"sql": sql})]


def _quoted(text: str) -> list[str]:
    return [p for i, p in enumerate(text.split("'")) if i % 2 == 1]
