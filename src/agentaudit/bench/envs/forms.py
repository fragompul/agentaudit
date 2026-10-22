"""Simulated web-forms environment (fill and submit with validation) and task family."""

from __future__ import annotations

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
from agentaudit.bench.words import CITIES, unique_people

FORM_DEFS: dict[str, dict[str, dict[str, object]]] = {
    "visitor-pass": {
        "full_name": {"required": True},
        "host": {"required": True},
        "visit_date": {"required": True, "format": "YYYY-MM-DD"},
        "badge_type": {"required": True, "options": ["V1", "V2", "V3"]},
    },
    "expense-claim": {
        "employee": {"required": True},
        "amount": {"required": True, "format": "number"},
        "category": {"required": True, "options": ["TRAVEL", "MEALS", "EQUIP"]},
        "city": {"required": False},
    },
    "it-request": {
        "requester": {"required": True},
        "device": {"required": True, "options": ["LAPTOP", "PHONE", "MONITOR"]},
        "urgency": {"required": True, "options": ["LOW", "HIGH"]},
    },
    "leave-request": {
        "employee": {"required": True},
        "start_date": {"required": True, "format": "YYYY-MM-DD"},
        "kind": {"required": True, "options": ["ANNUAL", "SICK"]},
    },
    "room-request": {
        "requester": {"required": True},
        "room": {"required": True, "options": ["R1", "R2", "R3"]},
        "date": {"required": True, "format": "YYYY-MM-DD"},
    },
}
OPTION_WORDS = {
    "V1": "day visitor",
    "V2": "contractor",
    "V3": "vip guest",
    "TRAVEL": "travel",
    "MEALS": "meals",
    "EQUIP": "equipment",
    "LAPTOP": "laptop",
    "PHONE": "phone",
    "MONITOR": "monitor",
    "LOW": "low",
    "HIGH": "high",
    "ANNUAL": "annual leave",
    "SICK": "sick leave",
    "R1": "small room",
    "R2": "medium room",
    "R3": "large room",
}


class FormsEnv(Environment):
    def __init__(self, form_ids: list[str]) -> None:
        super().__init__()
        self.form_ids = form_ids
        self.drafts: dict[str, dict[str, str]] = {f: {} for f in form_ids}
        self.submissions: list[dict[str, object]] = []
        self.register(ToolSpec("list_forms", "List the available forms."))
        fid = Param("string", "Form id")
        self.register(
            ToolSpec("get_form", "Show the fields of a form and their rules.", {"form_id": fid})
        )
        self.register(
            ToolSpec(
                "set_field",
                "Set the value of a field in a form draft.",
                {
                    "form_id": fid,
                    "field": Param("string", "Field name"),
                    "value": Param("string", "Value"),
                },
            )
        )
        self.register(
            ToolSpec("submit_form", "Validate and submit a form draft.", {"form_id": fid})
        )
        install_common_tools(self)

    def _form(self, args: JSONDict) -> tuple[str, dict[str, dict[str, object]]]:
        fid = str(args["form_id"])
        if fid not in self.form_ids:
            raise KeyError(f"unknown form {fid!r}")
        return fid, FORM_DEFS[fid]

    def _dispatch(self, name: str, args: JSONDict) -> str:
        common = handle_common(self, name, args)
        if common is not None:
            return common
        if name == "list_forms":
            return "\n".join(self.form_ids)
        fid, fields = self._form(args)
        if name == "get_form":
            lines = []
            for f, rule in fields.items():
                extra = ""
                if "options" in rule:
                    extra = " options=" + "/".join(rule["options"])  # type: ignore[arg-type]
                if "format" in rule:
                    extra += f" format={rule['format']}"
                lines.append(f"{f} ({'required' if rule['required'] else 'optional'}){extra}")
            return "\n".join(lines)
        if name == "set_field":
            f, v = str(args["field"]), str(args["value"])
            if f not in fields:
                return f"ERROR: unknown field {f!r}"
            opts = fields[f].get("options")
            if opts is not None and v not in opts:  # type: ignore[operator]
                return f"ERROR: invalid value for {f}; allowed: {'/'.join(opts)}"  # type: ignore[arg-type]
            self.drafts[fid][f] = v
            return f"Set {f}"
        if name == "submit_form":
            missing = [f for f, r in fields.items() if r["required"] and f not in self.drafts[fid]]
            if missing:
                return f"ERROR: missing required fields: {', '.join(missing)}"
            self.submissions.append({"form": fid, "values": dict(self.drafts[fid])})
            self.drafts[fid] = {}
            return "Submitted successfully."
        raise KeyError(name)

    def state(self) -> JSONDict:
        return {"submissions": self.submissions, "answer": self.answer}


class FormsFamily(Family):
    """Fill and submit ``min(3, horizon)`` forms; ``distractors`` extra forms are also listed."""

    name = "forms"
    NAME_FIELDS = ("full_name", "employee", "requester")
    DATE_FIELDS = ("visit_date", "start_date", "date")

    def _values(
        self, rng: np.random.Generator, fid: str, who: str, host: str
    ) -> tuple[dict[str, str], list[str]]:
        values: dict[str, str] = {}
        parts: list[str] = []
        for f, rule in FORM_DEFS[fid].items():
            if not rule["required"] and rng.random() < 0.5:
                continue
            label = f.replace("_", " ")
            if "options" in rule:
                code = str(rng.choice(rule["options"]))
                values[f] = code
                parts.append(f"{label}: {OPTION_WORDS[code]}")
                continue
            if f in self.DATE_FIELDS:
                values[f] = f"2026-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 28)):02d}"
            elif rule.get("format") == "number":
                values[f] = str(int(rng.integers(10, 900)))
            elif f == "host":
                values[f] = host
            elif f == "city":
                values[f] = str(rng.choice(CITIES))
            else:
                values[f] = who
            parts.append(f'{label}: "{values[f]}"')
        return values, parts

    def generate(
        self, rng: np.random.Generator, index: int, horizon: int, distractors: int
    ) -> Task:
        names = list(FORM_DEFS)
        n_jobs = min(3, max(1, horizon))
        start = index % len(names)
        job_ids = [names[(start + k) % len(names)] for k in range(n_jobs)]
        extras = [n for n in names if n not in job_ids][: max(0, distractors)]
        form_ids = [*job_ids, *extras]
        who, host = unique_people(rng, 2)
        jobs: list[JSONDict] = []
        sentences: list[str] = []
        for fid in job_ids:
            values, parts = self._values(rng, fid, who, host)
            jobs.append({"form": fid, "values": values})
            sentences.append(f"the '{fid}' form with: " + "; ".join(parts))
        instruction = (
            "Fill in and submit " + " and also ".join(sentences) + ". "
            "Option fields take a code; check each form for the allowed codes."
        )
        spec: JSONDict = {"form_ids": form_ids, "jobs": jobs}
        h = sum(len(j["values"]) + 2 for j in jobs)
        return Task(f"forms-{index:04d}", self.name, instruction, spec, h, distractors, False)

    def build(self, task: Task) -> FormsEnv:
        return FormsEnv(task.spec["form_ids"])

    def check(self, task: Task, env: Environment) -> bool:
        assert isinstance(env, FormsEnv)

        def key(x: JSONDict) -> str:
            return str(x["form"])

        return sorted(env.submissions, key=key) == sorted(task.spec["jobs"], key=key)

    def oracle(self, task: Task) -> list[ToolInvocation]:
        calls: list[ToolInvocation] = []
        for job in task.spec["jobs"]:
            fid = job["form"]
            calls.append(ToolInvocation("get_form", {"form_id": fid}))
            calls += [
                ToolInvocation("set_field", {"form_id": fid, "field": f, "value": v})
                for f, v in job["values"].items()
            ]
            calls.append(ToolInvocation("submit_form", {"form_id": fid}))
        return [*calls, ToolInvocation("finish", {})]
