from agentaudit.failures import CATEGORIES, classify, failure_table
from agentaudit.schema import Step, ToolCall, Trajectory


def traj(steps: list[Step], success: float = 0.0, term: str = "completed") -> Trajectory:
    return Trajectory("m", "t", 0, 0, success, tuple(steps), termination=term)


def call(name: str = "ls", ok: bool = True, err: str | None = None, **args: object) -> ToolCall:
    return ToolCall(name, dict(args), ok, err)


def test_success_is_none_and_categories_are_known() -> None:
    assert classify(traj([], 1.0)) is None
    assert "no_action" in CATEGORIES


def test_each_category() -> None:
    assert classify(traj([], term="error: BackendError")).category == "backend_error"  # type: ignore[union-attr]
    assert classify(traj([Step(0)])).category == "no_action"  # type: ignore[union-attr]
    bad = traj(
        [Step(0, tool_calls=(call(ok=False, err="ERROR: missing required argument(s): path"),))]
    )
    r = classify(bad)
    assert r is not None and r.category == "malformed_call" and r.step == 0
    transient = traj(
        [
            Step(
                0,
                tool_calls=(
                    call("write_file", ok=False, err="ERROR: transient failure (HTTP 503)"),
                ),
            )
        ]
    )
    assert classify(transient).category == "gave_up_after_transient_error"  # type: ignore[union-attr]
    loop = traj([Step(i, tool_calls=(call(path="/a"),)) for i in range(4)])
    assert classify(loop).category == "loop_or_step_limit"  # type: ignore[union-attr]
    limit = traj(
        [Step(0, tool_calls=(call(path="/a"),)), Step(1, tool_calls=(call(path="/b"),))],
        term="step_limit",
    )
    assert classify(limit).category == "loop_or_step_limit"  # type: ignore[union-attr]
    quit_early = traj([Step(0, tool_calls=(call("finish"),))])
    assert classify(quit_early).category == "premature_stop"  # type: ignore[union-attr]
    ans = traj(
        [
            Step(0, tool_calls=(call("query"),)),
            Step(1, tool_calls=(call("submit_answer", answer="3"),)),
        ]
    )
    assert classify(ans, True).category == "wrong_answer"  # type: ignore[union-attr]
    assert classify(traj([Step(0, tool_calls=(call("query"),))]), True).category == "premature_stop"  # type: ignore[union-attr]
    state = traj(
        [
            Step(0, tool_calls=(call("write_file", path="/a"),)),
            Step(1, tool_calls=(call("write_file", path="/b"),)),
            Step(2, tool_calls=(call("finish"),)),
        ]
    )
    assert classify(state).category == "wrong_final_state"  # type: ignore[union-attr]


def test_failure_table_counts() -> None:
    items = [traj([Step(0)]), traj([Step(0)]), traj([], 1.0)]
    tab = failure_table(items)
    assert tab["m"]["no_action"] == 2 and sum(tab["m"].values()) == 2
