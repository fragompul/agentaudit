"""Virtual file-system environment and task family."""

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
from agentaudit.bench.words import NOUNS, TOPICS, pick


class FileSystemEnv(Environment):
    def __init__(self, files: dict[str, str]) -> None:
        super().__init__()
        self.files = dict(files)
        self.dirs: set[str] = {"/"}
        for p in files:
            self._mk_parents(p)
        s = Param("string", "Absolute path, e.g. /docs/a.txt")
        self.register(ToolSpec("ls", "List the entries of a directory.", {"path": s}))
        self.register(ToolSpec("read_file", "Read a text file.", {"path": s}))
        self.register(
            ToolSpec(
                "write_file",
                "Create or overwrite a text file (parent directories are created).",
                {"path": s, "content": Param("string", "Full file content")},
            )
        )
        self.register(
            ToolSpec(
                "append_file",
                "Append text as a new line to an existing file.",
                {"path": s, "content": Param("string", "Line to append")},
            )
        )
        self.register(
            ToolSpec(
                "move",
                "Move or rename a file.",
                {"src": Param("string", "Source path"), "dst": Param("string", "Destination path")},
            )
        )
        self.register(ToolSpec("delete", "Delete a file.", {"path": s}))
        install_common_tools(self)

    def _mk_parents(self, path: str) -> None:
        parts = path.strip("/").split("/")[:-1]
        for i in range(1, len(parts) + 1):
            self.dirs.add("/" + "/".join(parts[:i]))

    @staticmethod
    def _norm(path: str) -> str:
        if not path.startswith("/"):
            raise ValueError("paths must be absolute (start with /)")
        return "/" + "/".join(p for p in path.split("/") if p)

    def _dispatch(self, name: str, args: JSONDict) -> str:
        common = handle_common(self, name, args)
        if common is not None:
            return common
        if name == "ls":
            d = self._norm(args["path"])
            if d not in self.dirs:
                return f"ERROR: no such directory: {d}"
            prefix = d.rstrip("/") + "/"
            names = set()
            for p in self.files:
                if p.startswith(prefix):
                    head, sep, _ = p[len(prefix) :].partition("/")
                    names.add(head + sep)
            for p in self.dirs:
                if p.startswith(prefix) and p != d:
                    names.add(p[len(prefix) :].split("/")[0] + "/")
            return "\n".join(sorted(names)) or "(empty)"
        if name == "read_file":
            p = self._norm(args["path"])
            return self.files.get(p, f"ERROR: no such file: {p}")
        if name == "write_file":
            p = self._norm(args["path"])
            self.files[p] = str(args["content"])
            self._mk_parents(p)
            return f"Wrote {len(self.files[p])} characters to {p}"
        if name == "append_file":
            p = self._norm(args["path"])
            if p not in self.files:
                return f"ERROR: no such file: {p}"
            sep = "" if self.files[p].endswith("\n") or not self.files[p] else "\n"
            self.files[p] += sep + str(args["content"])
            return f"Appended to {p}"
        if name == "move":
            a, b = self._norm(args["src"]), self._norm(args["dst"])
            if a not in self.files:
                return f"ERROR: no such file: {a}"
            if b in self.files:
                return f"ERROR: destination exists: {b}"
            self.files[b] = self.files.pop(a)
            self._mk_parents(b)
            return f"Moved {a} -> {b}"
        if name == "delete":
            p = self._norm(args["path"])
            if p not in self.files:
                return f"ERROR: no such file: {p}"
            del self.files[p]
            return f"Deleted {p}"
        raise KeyError(name)

    def state(self) -> JSONDict:
        return {"files": dict(sorted(self.files.items())), "answer": self.answer}


def _norm_text(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


class FileSystemFamily(Family):
    """Templates: create files, move by extension, append, find-and-answer, rename."""

    name = "fs"
    TEMPLATES = ("create", "move", "append", "find", "rename")

    def generate(
        self, rng: np.random.Generator, index: int, horizon: int, distractors: int
    ) -> Task:
        template = self.TEMPLATES[index % len(self.TEMPLATES)]
        topics = pick(rng, TOPICS, 3)
        nouns = pick(rng, NOUNS, 8)
        initial: dict[str, str] = {}
        for i in range(distractors):
            initial[f"/{topics[i % 3]}/{nouns[(i + 3) % 8]}_{i}.txt"] = (
                f"Unrelated {nouns[i % 8]} #{i}"
            )
        expect: dict[str, str] = {}
        absent: list[str] = []
        answer: str | None = None
        if template == "create":
            items = []
            for i in range(horizon):
                path = f"/{topics[0]}/{nouns[i]}.txt"
                content = f"{nouns[i]} for {topics[1]} number {int(rng.integers(10, 99))}"
                expect[path] = content
                items.append(f"'{path}' containing exactly the text: {content}")
            instruction = "Create the following text file(s): " + "; ".join(items) + "."
        elif template == "move":
            ext = str(rng.choice(["txt", "csv", "md"]))
            other = "log"
            for i in range(horizon):
                src = f"/inbox/{nouns[i]}.{ext}"
                initial[src] = f"data {i} {int(rng.integers(100, 999))}"
                expect[f"/archive/{nouns[i]}.{ext}"] = initial[src]
                absent.append(src)
            for i in range(distractors):
                initial[f"/inbox/{nouns[(i + 4) % 8]}_{i}.{other}"] = f"keep me {i}"
            instruction = (
                f"Move every .{ext} file from /inbox into /archive (keep the file names). "
                f"Do not touch files with other extensions."
            )
        elif template == "append":
            for i in range(horizon):
                path = f"/{topics[0]}/{nouns[i]}.txt"
                initial[path] = f"{nouns[i]} header"
                line = f"checked-{int(rng.integers(100, 999))}"
                expect[path] = f"{initial[path]}\n{line}"
            lines = [f"'{p}': {e.splitlines()[-1]}" for p, e in expect.items()]
            instruction = (
                "Append one new line to each of these files with exactly this text: "
                + "; ".join(lines)
                + "."
            )
        elif template == "find":
            token = f"CODE-{int(rng.integers(1000, 9999))}"
            target = int(rng.integers(0, max(horizon, 1)))
            for i in range(max(horizon, 2)):
                initial[f"/{topics[1]}/{nouns[i]}.txt"] = (
                    f"{nouns[i]} contains {token}" if i == target else f"{nouns[i]} plain text"
                )
            answer = f"{nouns[target]}.txt"
            instruction = (
                f"Exactly one file under /{topics[1]} mentions the code {token}. "
                f"Submit the file name (without directory) as the answer."
            )
        else:  # rename
            for i in range(horizon):
                initial[f"/{topics[2]}/{nouns[i]}.tmp"] = f"payload {i}-{int(rng.integers(10, 99))}"
                expect[f"/{topics[2]}/{nouns[i]}.final"] = initial[f"/{topics[2]}/{nouns[i]}.tmp"]
                absent.append(f"/{topics[2]}/{nouns[i]}.tmp")
            instruction = (
                f"Rename every .tmp file in /{topics[2]} to the same name "
                "with the .final extension."
            )
        spec: JSONDict = {
            "template": template,
            "initial": initial,
            "expect": expect,
            "absent": absent,
            "answer": answer,
        }
        return Task(
            f"fs-{index:04d}",
            self.name,
            instruction,
            spec,
            horizon=max(horizon, 2) if template == "find" else horizon,
            distractors=distractors,
            requires_answer=answer is not None,
        )

    def build(self, task: Task) -> FileSystemEnv:
        return FileSystemEnv(dict(task.spec["initial"]))

    def check(self, task: Task, env: Environment) -> bool:
        assert isinstance(env, FileSystemEnv)
        spec = task.spec
        if spec["answer"] is not None:
            return env.answer is not None and env.answer.strip().strip("'\"") == spec["answer"]
        for path, content in spec["expect"].items():
            if path not in env.files or _norm_text(env.files[path]) != _norm_text(content):
                return False
        if any(p in env.files for p in spec["absent"]):
            return False
        moved = set(spec["absent"]) | set(spec["expect"])
        return all(env.files.get(p) == c for p, c in spec["initial"].items() if p not in moved)

    def oracle(self, task: Task) -> list[ToolInvocation]:
        spec = task.spec
        calls: list[ToolInvocation] = []
        if spec["answer"] is not None:
            d = next(iter(spec["initial"])).rsplit("/", 1)[0]
            calls.append(ToolInvocation("ls", {"path": d}))
            return [*calls, ToolInvocation("submit_answer", {"answer": spec["answer"]})]
        t = spec["template"]
        for path, content in spec["expect"].items():
            if t == "create":
                calls.append(ToolInvocation("write_file", {"path": path, "content": content}))
            elif t == "append":
                calls.append(
                    ToolInvocation(
                        "append_file", {"path": path, "content": content.splitlines()[-1]}
                    )
                )
            else:
                src = next(
                    a
                    for a in spec["absent"]
                    if a.rsplit("/", 1)[1].split(".")[0] == path.rsplit("/", 1)[1].split(".")[0]
                )
                calls.append(ToolInvocation("move", {"src": src, "dst": path}))
        return [*calls, ToolInvocation("finish", {})]
