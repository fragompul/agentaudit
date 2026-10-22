"""Simulated REST-style CRM (customers and support tickets) and task family."""

from __future__ import annotations

import json
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
from agentaudit.bench.words import CITIES, FIRST, LAST

TIERS = ("free", "pro", "enterprise")
FIELDS = ("email", "tier", "city")


class CrmEnv(Environment):
    def __init__(self, customers: list[JSONDict], tickets: list[JSONDict]) -> None:
        super().__init__()
        self.customers = {int(c["id"]): dict(c) for c in customers}
        self.tickets = {int(t["id"]): dict(t) for t in tickets}
        self._next_ticket = max(self.tickets, default=100) + 1
        cid = Param("integer", "Customer id")
        self.register(
            ToolSpec(
                "search_customers",
                "Find customers whose name contains the text (case-insensitive).",
                {"name": Param("string", "Name or part of a name")},
            )
        )
        self.register(
            ToolSpec("get_customer", "Get a customer record by id.", {"customer_id": cid})
        )
        self.register(
            ToolSpec(
                "update_customer",
                "Set one field (email, tier or city) of a customer. "
                "tier is free, pro or enterprise.",
                {
                    "customer_id": cid,
                    "field": Param("string", "email, tier or city"),
                    "value": Param("string", "New value"),
                },
            )
        )
        self.register(
            ToolSpec(
                "create_ticket",
                "Open a support ticket for a customer.",
                {"customer_id": cid, "subject": Param("string", "Ticket subject")},
            )
        )
        self.register(ToolSpec("list_tickets", "List a customer's tickets.", {"customer_id": cid}))
        self.register(
            ToolSpec(
                "close_ticket",
                "Close a ticket by id.",
                {"ticket_id": Param("integer", "Ticket id")},
            )
        )
        install_common_tools(self)

    @staticmethod
    def _fmt(c: dict[str, Any]) -> str:
        return f"id={c['id']} name={c['name']} email={c['email']} tier={c['tier']} city={c['city']}"

    def _cust(self, args: JSONDict) -> dict[str, Any]:
        cid = int(args["customer_id"])
        if cid not in self.customers:
            raise KeyError(f"customer {cid} not found (404)")
        return self.customers[cid]

    def _dispatch(self, name: str, args: JSONDict) -> str:
        common = handle_common(self, name, args)
        if common is not None:
            return common
        if name == "search_customers":
            q = str(args["name"]).lower()
            hits = [c for c in self.customers.values() if q in str(c["name"]).lower()]
            return "\n".join(self._fmt(c) for c in hits[:10]) or "(no matches)"
        if name == "get_customer":
            return self._fmt(self._cust(args))
        if name == "update_customer":
            c = self._cust(args)
            field, value = str(args["field"]), str(args["value"])
            if field not in FIELDS:
                return f"ERROR: field must be one of {', '.join(FIELDS)}"
            if field == "tier" and value not in TIERS:
                return f"ERROR: tier must be one of {', '.join(TIERS)}"
            c[field] = value
            return "Updated. " + self._fmt(c)
        if name == "create_ticket":
            c = self._cust(args)
            tid = self._next_ticket
            self._next_ticket += 1
            self.tickets[tid] = {
                "id": tid,
                "customer_id": c["id"],
                "subject": str(args["subject"]),
                "status": "open",
            }
            return f"Created ticket {tid}"
        if name == "list_tickets":
            c = self._cust(args)
            rows = [t for t in self.tickets.values() if t["customer_id"] == c["id"]]
            return (
                "\n".join(f"ticket {t['id']}: {t['subject']} [{t['status']}]" for t in rows)
                or "(none)"
            )
        if name == "close_ticket":
            tid = int(args["ticket_id"])
            if tid not in self.tickets:
                return f"ERROR: ticket {tid} not found (404)"
            self.tickets[tid]["status"] = "closed"
            return f"Closed ticket {tid}"
        raise KeyError(name)

    def state(self) -> JSONDict:
        return {
            "customers": {str(k): v for k, v in sorted(self.customers.items())},
            "tickets": {str(k): v for k, v in sorted(self.tickets.items())},
            "answer": self.answer,
        }


class CrmFamily(Family):
    name = "crm"
    TEMPLATES = ("update_email", "open_ticket", "close_tickets", "upgrade_city", "lookup")

    def generate(
        self, rng: np.random.Generator, index: int, horizon: int, distractors: int
    ) -> Task:
        template = self.TEMPLATES[index % len(self.TEMPLATES)]
        n = 8 + 3 * distractors
        used: set[str] = set()
        customers: list[JSONDict] = []
        target_first, target_last = str(rng.choice(FIRST)), str(rng.choice(LAST))
        for i in range(n):
            # distractors share the target's last name (disambiguate by city)
            if 0 < i <= distractors:
                nm = f"{rng.choice(FIRST)} {target_last}"
            elif i == 0:
                nm = f"{target_first} {target_last}"
            else:
                nm = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            while nm in used:
                nm = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            used.add(nm)
            customers.append(
                {
                    "id": 1000 + i,
                    "name": nm,
                    "email": f"{nm.lower().replace(' ', '.')}@mail.test",
                    "tier": str(rng.choice(TIERS)),
                    "city": str(rng.choice(CITIES[:6])),
                }
            )
        # make the target's city unique among same-last-name customers
        tgt = customers[0]
        for c in customers[1 : distractors + 1]:
            while c["city"] == tgt["city"]:
                c["city"] = str(rng.choice(CITIES[:6]))
        tickets: list[JSONDict] = []
        answer: str | None = None
        expect_c = {c["id"]: dict(c) for c in customers}
        expect_t: dict[int, JSONDict] = {}
        if template == "update_email":
            new = f"{target_first.lower()}.new{int(rng.integers(10, 99))}@corp.test"
            expect_c[tgt["id"]]["email"] = new
            instruction = (
                f"Update the email of customer {tgt['name']} who lives in {tgt['city']} to {new}."
            )
        elif template == "open_ticket":
            subject = f"Refund request {int(rng.integers(100, 999))}"
            instruction = (
                f"Open a support ticket with subject '{subject}' for {tgt['name']} ({tgt['city']})."
            )
            expect_t[101] = {
                "id": 101,
                "customer_id": tgt["id"],
                "subject": subject,
                "status": "open",
            }
        elif template == "close_tickets":
            k = max(1, horizon)
            for j in range(k):
                tickets.append(
                    {
                        "id": 101 + j,
                        "customer_id": tgt["id"],
                        "subject": f"Issue {j}",
                        "status": "open",
                    }
                )
            tickets.append(
                {
                    "id": 150,
                    "customer_id": customers[-1]["id"],
                    "subject": "Other",
                    "status": "open",
                }
            )
            expect_t = {
                t["id"]: {**t, "status": "closed" if t["id"] != 150 else "open"} for t in tickets
            }
            instruction = f"Close every open ticket of customer {tgt['name']} ({tgt['city']})."
        elif template == "upgrade_city":
            city = str(tgt["city"])
            tgt["tier"] = expect_c[tgt["id"]]["tier"] = "free"  # guarantees a real change
            for c in expect_c.values():
                if c["city"] == city:
                    c["tier"] = "enterprise"
            instruction = f"Upgrade every customer located in {city} to the enterprise tier."
        else:
            answer = str(tgt["email"])
            instruction = (
                f"What is the email address of {tgt['name']} who lives in {tgt['city']}? "
                "Submit it as the answer."
            )
        spec: JSONDict = {
            "template": template,
            "customers": customers,
            "tickets": tickets,
            "expect_customers": list(expect_c.values()),
            "expect_tickets": list(expect_t.values()) or tickets,
            "answer": answer,
            "target": tgt["id"],
        }
        city_count = sum(1 for c in customers if c["city"] == tgt["city"])
        h = {
            "update_email": 2,
            "open_ticket": 2,
            "close_tickets": 2 + max(1, horizon),
            "upgrade_city": 1 + city_count,
            "lookup": 2,
        }[template]
        return Task(
            f"crm-{index:04d}", self.name, instruction, spec, h, distractors, answer is not None
        )

    def build(self, task: Task) -> CrmEnv:
        return CrmEnv(task.spec["customers"], task.spec["tickets"])

    def check(self, task: Task, env: Environment) -> bool:
        assert isinstance(env, CrmEnv)
        spec = task.spec
        if spec["answer"] is not None:
            return env.answer is not None and env.answer.strip() == spec["answer"]
        want_c = {int(c["id"]): c for c in spec["expect_customers"]}
        want_t = {int(t["id"]): t for t in spec["expect_tickets"]}
        got_t = {k: dict(t) for k, t in env.tickets.items()}
        return json.dumps(env.customers, sort_keys=True) == json.dumps(
            want_c, sort_keys=True
        ) and json.dumps(got_t, sort_keys=True) == json.dumps(want_t, sort_keys=True)

    def oracle(self, task: Task) -> list[ToolInvocation]:
        spec = task.spec
        tid = spec["target"]
        t = spec["template"]
        calls = [ToolInvocation("get_customer", {"customer_id": tid})]
        if t == "update_email":
            new = next(c for c in spec["expect_customers"] if c["id"] == tid)["email"]
            calls.append(
                ToolInvocation(
                    "update_customer", {"customer_id": tid, "field": "email", "value": new}
                )
            )
        elif t == "open_ticket":
            subj = spec["expect_tickets"][0]["subject"]
            calls.append(ToolInvocation("create_ticket", {"customer_id": tid, "subject": subj}))
        elif t == "close_tickets":
            calls += [
                ToolInvocation("close_ticket", {"ticket_id": x["id"]})
                for x in spec["tickets"]
                if x["customer_id"] == tid
            ]
        elif t == "upgrade_city":
            calls = [
                ToolInvocation(
                    "update_customer",
                    {"customer_id": c["id"], "field": "tier", "value": "enterprise"},
                )
                for c in spec["expect_customers"]
                if c["tier"] == "enterprise"
            ]
            # customers already enterprise are a no-op update, which is harmless
        else:
            calls.append(ToolInvocation("submit_answer", {"answer": spec["answer"]}))
        return [*calls, ToolInvocation("finish", {})]
