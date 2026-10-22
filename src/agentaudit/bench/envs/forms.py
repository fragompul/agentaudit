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
    name = "forms"

    def generate(
        self, rng: np.random.Generator, index: int, horizon: int, distractors: int
    ) -> Task:
        names = list(FORM_DEFS)
        fid = names[index % len(names)]
        form_ids = [fid, *[n for n in names if n != fid][: max(0, distractors)]]
        who, host = unique_people(rng, 2)
        values: dict[str, str] = {}
        parts: list[str] = []
        for f, rule in FORM_DEFS[fid].items():
            if not rule["required"] and rng.random() < 0.5:
                continue
            if "options" in rule:
                code = str(rng.choice(rule["options"]))
                values[f] = code
                parts.append(f"{f.replace('_', ' ')}: {OPTION_WORDS[code]}")
            elif rule.get("format") == "YYYY-MM-DD":
                values[f] = f"2026-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 28)):02d}"
                parts.append(f'{f.replace("_", " ")}: "{values[f]}"')
            elif rule.get("format") == "number":
                values[f] = str(int(rng.integers(10, 900)))
                parts.append(f'{f}: "{values[f]}"')
            elif f in ("host",):
                values[f] = host
                parts.append(f'{f}: "{host}"')
            elif f == "city":
                values[f] = str(rng.choice(CITIES))
                parts.append(f'{f}: "{values[f]}"')
            else:
                values[f] = who
                parts.append(f'{f.replace("_", " ")}: "{who}"')
        instruction = (
            f"Fill in and submit the '{fid}' form with: " + "; ".join(parts) + ". "
            "Option fields take a code; check the form for the allowed codes."
        )
        spec: JSONDict = {"form_ids": form_ids, "form": fid, "values": values}
        return Task(
            f"forms-{index:04d}", self.name, instruction, spec, len(values) + 2, distractors, False
        )

    def build(self, task: Task) -> FormsEnv:
        return FormsEnv(task.spec["form_ids"])

    def check(self, task: Task, env: Environment) -> bool:
        assert isinstance(env, FormsEnv)
        want = {"form": task.spec["form"], "values": task.spec["values"]}
        return env.submissions == [want]

    def oracle(self, task: Task) -> list[ToolInvocation]:
        fid = task.spec["form"]
        calls = [ToolInvocation("get_form", {"form_id": fid})]
        calls += [
            ToolInvocation("set_field", {"form_id": fid, "field": f, "value": v})
            for f, v in task.spec["values"].items()
        ]
        return [
            *calls,
            ToolInvocation("submit_form", {"form_id": fid}),
            ToolInvocation("finish", {}),
        ]
