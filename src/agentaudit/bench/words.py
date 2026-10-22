"""Small fixed vocabularies for procedural task generation (deterministic, offline)."""

from __future__ import annotations

import numpy as np

FIRST = [
    "Alice",
    "Bruno",
    "Carla",
    "Diego",
    "Elena",
    "Farid",
    "Gina",
    "Hugo",
    "Irene",
    "Jonas",
    "Karla",
    "Liam",
    "Marta",
    "Noah",
    "Olga",
    "Pablo",
    "Quinn",
    "Rosa",
    "Samir",
    "Tania",
    "Ulises",
    "Vera",
    "Walter",
    "Ximena",
    "Yara",
    "Zane",
    "Ana",
    "Bea",
    "Cleo",
    "Dario",
]
LAST = [
    "Alvarez",
    "Brandt",
    "Castro",
    "Duarte",
    "Esposito",
    "Fischer",
    "Garcia",
    "Haddad",
    "Ibarra",
    "Jensen",
    "Klein",
    "Lopez",
    "Moreau",
    "Novak",
    "Ortega",
    "Petrov",
    "Quispe",
    "Ramos",
    "Silva",
    "Tanaka",
    "Urbina",
    "Vidal",
    "Weber",
    "Xu",
    "Yilmaz",
    "Zamora",
]
NOUNS = [
    "report",
    "invoice",
    "budget",
    "summary",
    "notes",
    "draft",
    "plan",
    "memo",
    "ledger",
    "roster",
    "agenda",
    "policy",
    "contract",
    "survey",
    "backup",
    "log",
    "manifest",
    "receipt",
    "schedule",
    "proposal",
]
TOPICS = [
    "sales",
    "hr",
    "legal",
    "ops",
    "design",
    "infra",
    "security",
    "research",
    "support",
    "finance",
]
CITIES = [
    "Sevilla",
    "Madrid",
    "Lisbon",
    "Porto",
    "Paris",
    "Berlin",
    "Vienna",
    "Rome",
    "Prague",
    "Dublin",
    "Oslo",
    "Athens",
]
PRODUCTS = [
    "Widget",
    "Gadget",
    "Sprocket",
    "Gizmo",
    "Flange",
    "Bracket",
    "Valve",
    "Sensor",
    "Relay",
    "Gasket",
]
COLORS = ["red", "blue", "green", "amber", "violet", "teal", "coral", "ivory"]


def person(rng: np.random.Generator) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def unique_people(rng: np.random.Generator, n: int) -> list[str]:
    seen: list[str] = []
    while len(seen) < n:
        p = person(rng)
        if p not in seen:
            seen.append(p)
    return seen


def pick(rng: np.random.Generator, pool: list[str] | tuple[str, ...], n: int) -> list[str]:
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[int(i)] for i in idx]
