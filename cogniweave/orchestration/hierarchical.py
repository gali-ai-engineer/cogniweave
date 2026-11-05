"""Hierarchical orchestration: supervisor → planner → parallel DAG → critic loop.

1. **Plan** – the planner emits a JSON DAG. It is validated (schema, unknown
   roles, cycles); invalid plans are sent back with the error for self-repair,
   with a safe fallback plan after ``max_plan_attempts``.
2. **Execute** – :class:`DAGExecutor` runs independent tasks concurrently.
3. **Review ⇄ revise** – the critic scores the final deliverable; below the
   threshold the writer *reflects* (Reflexion), stores a lesson, and revises.
4. **Learn** – the writer's strategy is credited with the final review score.
"""

from __future__ import annotations

from ..crew import Crew
from .executor import DAGExecutor, TaskResult
from .plan import Plan, PlanTask, Review, extract_json, revision_log, stalled
from .result import RunResult


class HierarchicalOrchestrator:
    def __init__(
        self,
        crew: Crew,
        *,
        max_concurrency: int = 4,
        max_revisions: int = 2,
        accept_threshold: float = 0.8,
        max_plan_attempts: int = 3,
        low_confidence: float = 0.5,
    ) -> None:
        self.crew = crew
        self.tracer = crew.tracer
        self.max_revisions = max_revisions
        self.accept_threshold = accept_threshold
        self.max_plan_attempts = max_plan_attempts
        self.worker_roles = [r for r in crew.roles if r not in ("planner", "critic")]
        self.executor = DAGExecutor(
            crew.agents,
            tracer=crew.tracer,
            blackboard=crew.blackboard,
            max_concurrency=max_concurrency,
            low_confidence=low_confidence,
        )

    # ------------------------------------------------------------- planning
    def _validate(self, data: object, goal: str) -> Plan:
        if isinstance(data, dict):
            data.setdefault("goal", goal)
        plan = Plan.from_dict(data)
        unknown = {t.role for t in plan.tasks} - set(self.worker_roles)
        if unknown:
            raise ValueError(f"unknown roles {sorted(unknown)}; allowed: {self.worker_roles}")
        return plan

    def fallback_plan(self, goal: str) -> Plan:
        return Plan(
            goal=goal,
            tasks=[
                PlanTask(id="research", role="researcher", description=f"Research: {goal}"),
                PlanTask(
                    id="write",
                    role="writer",
                    description="Write a cited report answering the goal",
                    depends_on=["research"],
                ),
            ],
        )

    async def make_plan(self, goal: str) -> Plan:
        planner = self.crew["planner"]
        task = f"Create an execution plan. Allowed roles: {', '.join(self.worker_roles)}."
        feedback = ""
        async with self.tracer.span("plan", goal=goal):
            for attempt in range(1, self.max_plan_attempts + 1):
                out = await planner.run(task, goal=goal, feedback=feedback)
                try:
                    plan = self._validate(extract_json(out.content), goal)
                    self.tracer.event(
                        "plan.accepted", kind="note", text=f"plan with {len(plan.tasks)} tasks", attempt=attempt
                    )
                    await self.crew.blackboard.post("plan", plan.to_json(), planner.name)
                    return plan
                except Exception as exc:  # self-repair: feed the validation error back
                    feedback = f"Your previous plan was invalid: {exc}. Return corrected JSON only."
                    self.tracer.event("plan.rejected", kind="note", text=f"plan invalid, attempt {attempt}")
            self.tracer.event("plan.fallback", kind="note", text="using fallback plan")
            return self.fallback_plan(goal)

    # ------------------------------------------------------------- phases
    async def execute(self, plan: Plan) -> dict[str, TaskResult]:
        return await self.executor.run(plan, deferred_reward={t.id for t in plan.sinks()})

    async def review(self, goal: str, draft: str) -> Review:
        out = await self.crew["critic"].run("Review the draft report against the goal.", goal=goal, context=draft)
        review = Review.parse(out.content)
        await self.crew.blackboard.post(
            "critique", review.to_json(), "critic", confidence=review.score, tags=(review.verdict,)
        )
        return review

    async def revise(self, final: TaskResult, goal: str, review: Review) -> str:
        agent = self.crew[final.task.role]
        feedback = review.to_json()
        await agent.reflect(final.task.description, feedback)
        out = await agent.run(final.task.description, goal=goal, context=final.context, feedback=feedback)
        final.output = out
        await self.crew.blackboard.post("draft", out.content, agent.name, confidence=out.confidence)
        return out.content

    # ------------------------------------------------------------- run
    async def run(self, goal: str) -> RunResult:
        async with self.tracer.span("hierarchical.run", goal=goal):
            await self.crew.blackboard.post("goal", goal, "user")
            plan = await self.make_plan(goal)
            results = await self.execute(plan)
            final = results[plan.sinks()[-1].id]
            draft, review, revisions, reviews = final.content, None, 0, []
            if final.ok and "critic" in self.crew.agents:
                review = await self.review(goal, draft)
                reviews.append(review)
                while review.score < self.accept_threshold and revisions < self.max_revisions:
                    revisions += 1
                    draft = await self.revise(final, goal, review)
                    review = await self.review(goal, draft)
                    reviews.append(review)
                    if stalled(revision_log(reviews)):  # a rewording that fixed nothing: stop, don't loop
                        self.tracer.event("revision.stalled", kind="note", text="revision fixed nothing, stopping")
                        break
                if final.output:
                    self.crew[final.task.role].learn(final.output.strategy, review.score)
        return RunResult(
            "hierarchical",
            goal,
            draft,
            self.tracer,
            review=review,
            plan=plan,
            revisions=revisions,
            blackboard=self.crew.blackboard.snapshot(),
            extra={
                "task_results": {k: v.content for k, v in results.items()},
                "follow_ups": [k for k, v in results.items() if v.followed_up],
                "revision_log": revision_log(reviews),
                "unresolved_issues": review.issues if review and review.score < self.accept_threshold else [],
            },
        )
