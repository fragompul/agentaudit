# AgentAudit-Bench

Six simulated environments, all offline and deterministic, with tasks generated procedurally from a seed.

| family | world | tools | task templates |
|---|---|---|---|
| `fs` | virtual file system | ls, read_file, write_file, append_file, move, delete | create, move by extension, append, find-and-answer, rename |
| `sql` | in-memory SQLite | list_tables, describe, query, execute | count, revenue, raise, delete, insert |
| `crm` | REST-style customers and tickets | search/get/update customer, create/list/close ticket | update email, open ticket, close tickets, upgrade by city, lookup |
| `booking` | room reservations | list_free_slots, book_slot, cancel_booking, list_bookings | book first free, book given, cancel all, move, count free |
| `forms` | web forms with validation | list_forms, get_form, set_field, submit_form | fill and submit 1-3 forms |
| `tabular` | table + calculator | list_tables, show_rows, calc | sum, revenue, max, average, difference |

All environments also expose `submit_answer` and `finish`.

## Difficulty knobs

`generate_suite(n_per_family=60)` sweeps a factorial grid of **horizon** (1-4 operations) and **distractors** (0-2: confusable
files, decoy tables, same-surname customers, extra forms, extra rows). Templates cycle with the task index. The
*core-120* subset used for local-model evaluations keeps every third task of each family.

## Verification

Success is decided by the **final world state** (files, rows, tickets, bookings, submitted forms) or the submitted answer, never by the
agent's text. `agentaudit.bench.audit` enforces three properties for every task: the reference **oracle** solves it, a **do-nothing agent** does
not, and generation and the resulting state hash are **deterministic**. Free-text values in instructions are quoted to avoid
punctuation ambiguity.

## Controlled stochasticity

`FaultInjector` fails each tool call independently with probability `rate` using a generator seeded by `(task, run)`, so every model
sees the **same** failures for the same attempt; failed calls leave the world unchanged and return a transient 503 message. Injections are logged.

## Agents

* `TextProtocolAgent`: a uniform ReAct-style protocol over any chat endpoint (Ollama, llama.cpp, mlx_lm.server, LM Studio). The benchmark defines
  how tools are described and parsed. A call is any JSON object with `name` and `arguments`; only the first call per turn is executed, and
  generation stops after it.
* `OllamaAgent` / `OpenAICompatAgent`: native tool-calling APIs. *Note*: Ollama 0.40.8 silently dropped well-formed calls for qwen2.5:3b with eight declared tools,
  which is why experiments use the text protocol.
* `SkillAgent`, `OracleAgent`, `NullAgent`: synthetic agents.
* `CachedAgent`: record/replay cache keyed by agent identity, history, tools and seed.
