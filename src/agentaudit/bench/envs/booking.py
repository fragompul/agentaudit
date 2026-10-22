"""Room/slot reservation environment and task family."""

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
from agentaudit.bench.words import unique_people

HOURS = (9, 10, 11, 12, 14, 15, 16, 17)
ROOMS = ("Aurora", "Borealis", "Cirrus", "Delta")


def slot_id(day: int, room: str, hour: int) -> str:
    return f"D{day}-{room[0]}{hour:02d}"


class BookingEnv(Environment):
    def __init__(self, booked: dict[str, str], days: int) -> None:
        super().__init__()
        self.days = days
        self.slots: dict[str, dict[str, object]] = {}
        for d in range(1, days + 1):
            for r in ROOMS:
                for h in HOURS:
                    sid = slot_id(d, r, h)
                    self.slots[sid] = {"day": d, "room": r, "hour": h, "by": booked.get(sid)}
        self.register(
            ToolSpec(
                "list_free_slots",
                "List free slots on a day (1-based), optionally for one room.",
                {
                    "day": Param("integer", "Day number"),
                    "room": Param("string", "Room name", required=False),
                },
            )
        )
        slot = Param("string", "Slot id such as D1-A09")
        self.register(
            ToolSpec(
                "book_slot",
                "Book a free slot for a person.",
                {"slot_id": slot, "name": Param("string", "Person")},
            )
        )
        self.register(ToolSpec("cancel_booking", "Cancel an existing booking.", {"slot_id": slot}))
        self.register(
            ToolSpec(
                "list_bookings",
                "List all bookings of a person.",
                {"name": Param("string", "Person")},
            )
        )
        install_common_tools(self)

    def _dispatch(self, name: str, args: JSONDict) -> str:
        common = handle_common(self, name, args)
        if common is not None:
            return common
        if name == "list_free_slots":
            day = int(args["day"])
            room = args.get("room")
            out = [
                f"{k} ({v['room']} {v['hour']}:00)"
                for k, v in self.slots.items()
                if v["day"] == day and v["by"] is None and (room is None or v["room"] == room)
            ]
            return "\n".join(out) or "(no free slots)"
        if name == "book_slot":
            s = self.slots.get(str(args["slot_id"]))
            if s is None:
                return "ERROR: unknown slot"
            if s["by"] is not None:
                return f"ERROR: slot already booked by {s['by']}"
            s["by"] = str(args["name"])
            return f"Booked {args['slot_id']} for {args['name']}"
        if name == "cancel_booking":
            s = self.slots.get(str(args["slot_id"]))
            if s is None:
                return "ERROR: unknown slot"
            if s["by"] is None:
                return "ERROR: slot is not booked"
            s["by"] = None
            return f"Cancelled {args['slot_id']}"
        if name == "list_bookings":
            who = str(args["name"])
            out = [
                f"{k} ({v['room']} day {v['day']} {v['hour']}:00)"
                for k, v in self.slots.items()
                if v["by"] == who
            ]
            return "\n".join(out) or "(no bookings)"
        raise KeyError(name)

    def bookings(self) -> dict[str, str]:
        return {k: str(v["by"]) for k, v in self.slots.items() if v["by"] is not None}

    def state(self) -> JSONDict:
        return {"bookings": dict(sorted(self.bookings().items())), "answer": self.answer}


class BookingFamily(Family):
    name = "booking"
    TEMPLATES = ("book_first", "book_room_hour", "cancel_all", "move", "count_free")

    def generate(
        self, rng: np.random.Generator, index: int, horizon: int, distractors: int
    ) -> Task:
        template = self.TEMPLATES[index % len(self.TEMPLATES)]
        days = 2
        who, other = unique_people(rng, 2)
        all_ids = [slot_id(d, r, h) for d in range(1, days + 1) for r in ROOMS for h in HOURS]
        busy = [
            str(x)
            for x in rng.choice(
                all_ids, size=min(len(all_ids), 12 + 6 * distractors), replace=False
            )
        ]
        booked = dict.fromkeys(busy, other)
        expect = dict(booked)
        answer: str | None = None
        if template == "book_first":
            day = int(rng.integers(1, days + 1))
            room = str(rng.choice(ROOMS))
            free = [slot_id(day, room, h) for h in HOURS if slot_id(day, room, h) not in booked]
            if not free:
                booked.pop(slot_id(day, room, HOURS[0]), None)
                expect = dict(booked)
                free = [slot_id(day, room, h) for h in HOURS if slot_id(day, room, h) not in booked]
            expect[free[0]] = who
            instruction = f"Book the earliest free slot in room {room} on day {day} for {who}."
        elif template == "book_room_hour":
            items = []
            for _ in range(max(1, horizon)):
                sid = next(s for s in rng.permutation(all_ids) if s not in expect)
                expect[str(sid)] = who
                d = int(str(sid)[1])
                items.append(f"{sid} (day {d})")
            instruction = f"Book these slots for {who}: {', '.join(items)}."
        elif template == "cancel_all":
            mine = [
                str(s)
                for s in rng.choice(
                    [s for s in all_ids if s not in booked], size=max(1, horizon), replace=False
                )
            ]
            for s in mine:
                booked[s] = who
                del_expect = expect.pop(s, None)
                assert del_expect is None
            instruction = f"Cancel all bookings of {who}. Do not touch anyone else's bookings."
        elif template == "move":
            old = str(rng.choice([s for s in all_ids if s not in booked]))
            booked[old] = who
            new = str(next(s for s in rng.permutation(all_ids) if s not in booked and s != old))
            expect = {k: v for k, v in booked.items() if k != old} | {new: who}
            instruction = (
                f"{who} has one booking. Move it to slot {new}: "
                f"cancel the old booking and book {new}."
            )
        else:
            day = int(rng.integers(1, days + 1))
            room = str(rng.choice(ROOMS))
            n_free = sum(1 for h in HOURS if slot_id(day, room, h) not in booked)
            answer = str(n_free)
            instruction = (
                f"How many free slots does room {room} have on day {day}? "
                "Submit the number as the answer."
            )
        spec: JSONDict = {
            "template": template,
            "days": days,
            "booked": booked,
            "expect": expect,
            "answer": answer,
            "who": who,
        }
        diff = sum(1 for k in set(booked) | set(expect) if booked.get(k) != expect.get(k))
        h = (
            1
            if template == "count_free"
            else max(1, diff) + (1 if template in ("book_first", "move", "cancel_all") else 0)
        )
        return Task(
            f"booking-{index:04d}", self.name, instruction, spec, h, distractors, answer is not None
        )

    def build(self, task: Task) -> BookingEnv:
        return BookingEnv(dict(task.spec["booked"]), task.spec["days"])

    def check(self, task: Task, env: Environment) -> bool:
        assert isinstance(env, BookingEnv)
        spec = task.spec
        if spec["answer"] is not None:
            return env.answer is not None and env.answer.strip() == spec["answer"]
        return bool(env.bookings() == spec["expect"])

    def oracle(self, task: Task) -> list[ToolInvocation]:
        spec = task.spec
        if spec["answer"] is not None:
            return [ToolInvocation("submit_answer", {"answer": spec["answer"]})]
        calls = [
            ToolInvocation("cancel_booking", {"slot_id": k})
            for k in spec["booked"]
            if k not in spec["expect"]
        ]
        calls += [
            ToolInvocation("book_slot", {"slot_id": k, "name": v})
            for k, v in spec["expect"].items()
            if spec["booked"].get(k) != v
        ]
        return [*calls, ToolInvocation("finish", {})]
