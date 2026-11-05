# Evaluating and operating agent systems

Agent systems should be evaluated end to end on a fixed task set, measuring answer quality alongside cost, latency and the number of model calls. A design that improves quality slightly while tripling cost may not be worth shipping.

Observability is essential for multi-agent systems. Tracing every agent run, tool call and model call with timings and token counts makes it possible to see where time and money go and to debug failures after the fact.

Deterministic offline tests catch regressions in orchestration logic without paying for model calls. A mock model that follows the same protocols as a real one lets planning, tool use and review loops run in continuous integration.

Guardrails should bound every run. Budgets on model calls and tokens, timeouts, retries with exponential backoff and a maximum number of revision rounds prevent runaway loops and surprise bills.

Groundedness can be checked automatically. Requiring each claim to carry a source citation, and having a critic reject drafts without a sources section, makes outputs auditable and reduces unsupported statements.

Provider independence reduces risk. Keeping the orchestration code behind a small model interface means the same system can run on hosted models or local models, and can switch providers when price or quality changes.
