# Adding an agent or an environment

## An agent

Subclass `agentaudit.agents.base.Agent`, set `name`, and implement `act(messages, tools, *, seed) -> AgentAction`. Keep per-attempt state on the
object and initialise it in `begin(task, family, seed)`: the runner calls `session()`, which shallow-copies the agent per attempt so attempts can
run concurrently. Wrap it in `CachedAgent` to record and replay.

## An environment

1. Subclass `Environment`, register `ToolSpec`s and implement `_dispatch` and `state`. Tool failures must be returned as `"ERROR: ..."` strings.
2. Subclass `Family`: `generate` (deterministic in the supplied generator), `build`, `check` (final state only), `oracle`.
3. Register it in `agentaudit.bench.suite.FAMILIES`.
4. Run `agentaudit bench audit`. A family is accepted only if the oracle always passes, the null agent never does, and the state hash is deterministic.
