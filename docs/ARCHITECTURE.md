# Architecture notes

This document explains the mechanisms behind `cogniweave` in more depth than the README, including the trade-offs behind each choice.

## 1. Layers

| Layer | Responsibility | Depends on |
|---|---|---|
| `llm` | One async `complete()` call per provider; token usage in the response | nothing |
| `cognition` | Embeddings and the four memory systems | `llm.base` (token estimate) |
| `core` | `CognitiveAgent`, tools, blackboard, tracing | `cognition`, `llm` |
| `orchestration` | Plans, DAG executor, four patterns, `RunResult` | `core`, `crew` |
| `adapters` | Third-party control-flow backends (LangGraph) | `orchestration` |

Dependencies only point downward, so any layer can be reused alone — e.g. a single `CognitiveAgent` with custom tools needs no orchestrator.

## 2. The agent

`CognitiveAgent.run()` is one pass of a perceive → recall → deliberate/act → learn cycle.

**Perceive.** A fresh `WorkingMemory` is created per run with a token budget (default 3,000). Context from upstream tasks enters with salience 1.0.

**Recall.** `EpisodicMemory.recall(task, agent=self.name)` returns the top episodes by

```
score = 1.0·cosine(query, episode) + 0.5·0.5^(age / half_life) + 0.7·importance
```

Episodes below a minimum relevance are excluded before ranking, so a high-importance but unrelated memory cannot be injected. Reflections (importance 0.9) become `<lessons>`; the single best past action becomes `<past_experience>` at low salience, so it is the first thing evicted under pressure.

`ProceduralMemory.select(role, strategies)` picks a strategy with UCB1: every strategy is tried once, then `argmax(mean + c·sqrt(ln N / n))`. Rewards are the agent's confidence or, for reviewed deliverables, the critic's score. Rewards for the final deliverable are *deferred* until review so the writer is credited once, with the external signal.

**Deliberate ⇄ act.** A ReAct loop: if the reply contains `ACTION: <tool> <json>`, the tool runs and its output comes back as `OBSERVATION:`. Tool errors are returned as observations. On the last allowed step the agent is told to answer without tools.

**Learn.** The outcome is stored as an episode whose importance rises as confidence falls — uncertain outcomes are more worth remembering.

**Reflect.** Called by an orchestrator when a critic rejects work. The agent turns the critique into one reusable `LESSON:` stored with importance 0.9. Because the lesson is retrieved by relevance to the *task*, it applies to future similar tasks, including in later processes when memory is persisted.

## 3. Plans and execution

`Plan` is a Pydantic model whose validator rejects duplicate ids, unknown dependencies, self-dependencies and cycles (Kahn's algorithm). `HierarchicalOrchestrator` additionally rejects roles that are not in the crew. On any validation error the planner receives the error text as `<feedback>` and tries again (default three attempts) before a safe two-task fallback plan is used.

`DAGExecutor` starts a coroutine per task. Each waits on its dependencies' `asyncio.Event`s, then acquires a semaphore (default concurrency 4) to call its agent. This starts every task at the earliest possible moment — better than wave-by-wave execution when task durations differ. Failures are retried; a failed task still releases its dependents, which receive "(task failed: …)" in their context rather than hanging.

**Metacognitive follow-up:** if a researcher returns confidence below 0.5, it is asked once to deepen the investigation and the two outputs are merged.

## 4. Blackboard control

Each `KnowledgeSource` implements `activation(board) -> float` and `contribute(board, goal)`:

| Source | Fires when | Activation |
|---|---|---|
| Decomposer | no questions yet | 1.0 |
| Researcher | some question has no finding referencing it | 0.9 |
| Analyst | all questions answered and analysis is missing or stale | 0.8 |
| Critic | the latest draft has no critique | 0.75 |
| Writer | analysis exists and there is no draft, or the latest critique says "revise" | 0.7 |

The controller loop runs the highest positive activation until a critique scores above the threshold, nothing is activated, or `max_cycles` is reached. Entries carry `refs` to the entries they build on, giving a provenance graph for every answer.

Trade-off: the blackboard adapts well (e.g. a new question posted mid-run is researched automatically) but its schedule is less predictable than a plan; tracing every scheduling decision is what keeps it debuggable.

## 5. Debate

Debaters with different personas share one retrieved evidence set (so disagreement comes from reasoning, not different inputs). In round *n > 1* each sees the others' latest positions in `<feedback>` and must add a rebuttal. A judge aggregates the final round. Cost grows as `rounds × debaters + 1` calls, which is why debate suits contested questions rather than routine tasks.

## 6. Tracing and budgets

`Tracer.span()` stores the parent span id in a `ContextVar`; `asyncio.gather` copies context into each task, so spans from concurrent agents nest under the correct parent without explicit plumbing. `check_budget()` runs before every LLM call. Exports: `to_dict()/save_json()` for tooling and `to_mermaid()` for humans.

## 7. Testing strategy

The mock LLM implements every role's protocol deterministically, which lets the tests assert *behaviour* rather than text: the exact blackboard schedule, that a bad plan is repaired from feedback, that low confidence triggers follow-ups, that a lesson removes a revision in the next run, that a budget stops a run, and — using simulated latency — that independent tasks actually run in parallel.

## 8. Known limitations

- The default hashing embedder is lexical; swap in a neural embedder for semantic recall on paraphrases.
- Episodic memory is unbounded; consolidation is on the roadmap.
- The mock is protocol-faithful, not intelligent — quality claims must be measured with real providers.
