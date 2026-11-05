# Orchestration patterns

In a hierarchical or supervisor pattern, a planner decomposes the goal into a graph of tasks and dispatches them to specialist workers. Representing the plan as a directed acyclic graph lets independent tasks run in parallel and makes the dependencies explicit and checkable.

Plans produced by a language model can be invalid, with unknown roles, missing dependencies or cycles. Validating the plan and feeding the error back to the planner for self-repair, with a safe fallback plan, keeps the system robust.

The blackboard pattern, first used in the Hearsay-II speech understanding system, coordinates independent knowledge sources through a shared workspace. Each source watches the board and contributes when its preconditions are met, so control emerges from the data rather than from a fixed script.

Blackboard systems are flexible and handle unexpected situations well, but their behaviour is harder to predict and debug because the order of contributions depends on the evolving state of the board.

Multi-agent debate has several agents answer independently, read each other's answers and revise over multiple rounds before a judge aggregates them. Debate can improve factuality and reasoning on contested questions, but it multiplies the number of model calls.

A sequential pipeline passes the output of one agent directly to the next. It is the simplest pattern to reason about and a useful baseline, but it cannot exploit parallelism and has no built-in mechanism to catch errors.

Critic or reviewer loops add a quality gate before an answer is returned. A critic scores the draft against explicit criteria and the author revises until the score passes a threshold or a revision limit is reached.
