# Security policy

AgentAudit is a research/evaluation library; it executes no model-generated code outside the
sandboxed, in-memory simulated environments of AgentAudit-Bench (a virtual file system, an in-memory
SQLite database and pure-Python simulators; the calculator uses a restricted AST evaluator).

If you find a vulnerability (for example a way for an agent to escape a simulated environment, or an
unsafe deserialisation path), please report it privately through GitHub's
[security advisories](https://github.com/fragompul/agentaudit/security/advisories/new) rather than a
public issue. Expect an acknowledgement within a week.
