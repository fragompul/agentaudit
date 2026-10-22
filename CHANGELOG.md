# Changelog

All notable changes are documented here ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/), [SemVer](https://semver.org/)).

## [0.1.0] - 2026-10-23

### Added
- Statistics: unbiased pass@k and pass^k; Wilson, Clopper-Pearson, Beta-Binomial and profile-likelihood
  intervals; task / run / two-stage and BCa bootstrap; crossed variance decomposition with generalizability
  coefficients; paired tests (exact McNemar, cluster bootstrap, sign-flip) with Holm and Benjamini-Hochberg;
  rank-uncertainty distributions; analytic and simulated power planning with optimal task/run allocation;
  anytime-valid betting confidence sequences.
- AgentAudit-Bench: six offline deterministic environments (virtual file system, SQLite, CRM REST, bookings,
  web forms, tabular + calculator), 360 procedurally generated tasks, state-based checkers, seeded fault
  injection, oracle/null self-audit.
- Agents: uniform text tool protocol, Ollama and OpenAI-compatible local backends, synthetic skill agents,
  record/replay cache. Resumable run-major runner.
- Ingestion adapters for tau-bench and SWE-bench; failure taxonomy; Markdown/HTML reports; CLI; web playground.
