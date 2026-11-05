# Multi-agent LLM systems

A multi-agent LLM system splits a problem across several model-driven agents, each with its own role, prompt, tools and memory. Specialisation lets each agent focus on a narrow job, which tends to improve output quality on tasks that decompose cleanly into research, analysis and writing.

Parallelism is a major benefit of multi-agent designs. Independent subtasks such as researching separate questions can run concurrently, which reduces wall-clock latency even though the total number of model calls goes up.

Role separation also improves controllability. When a planner, a worker and a reviewer are distinct agents, each step produces an inspectable artefact, so failures can be traced to a specific agent instead of a single opaque prompt.

Multi-agent systems carry a real coordination cost. Every hand-off adds tokens, latency and an opportunity for information to be lost or distorted between agents, so a team of agents can be slower and more expensive than one well-prompted model.

Errors can compound across agents. If an early researcher returns a wrong claim, downstream analysts and writers may amplify it with confident language, which is why grounding claims in cited sources and adding a critic agent matter.

Teams of agents can also fall into unproductive loops, where agents keep revising or delegating without converging. Production systems therefore need explicit budgets on calls and tokens, termination conditions, and a maximum number of revision rounds.

Single-agent systems are simpler to build, debug and evaluate. A single agent with good tools and retrieval is often the right starting point, and teams should add agents only when specialisation or parallelism clearly pays off on measured evaluations.
