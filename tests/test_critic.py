"""The critic must catch substance problems, and a reworded draft must not pass."""

import asyncio
import json
import re

from cogniweave import BlackboardOrchestrator, MockLLM, build_research_crew, run_pattern
from cogniweave.cognition.memory import SemanticMemory
from cogniweave.llm.base import Message
from cogniweave.orchestration.plan import Review, revision_log, stalled

QUESTION = "When is a blackboard architecture better than a fixed plan?"


def critique(draft: str, goal: str = QUESTION) -> dict:
    prompt = f"<role>critic</role>\n<goal>{goal}</goal>\n<task>Review</task>\n<context>\n{draft}\n</context>"
    resp = asyncio.run(MockLLM().complete([Message("user", prompt)], metadata={"role": "critic"}))
    return json.loads(resp.text)


GOOD = """# Report
## Key findings
- The blackboard pattern coordinates sources through a shared workspace. [orchestration_patterns.md#2]
- Blackboard control emerges from the data rather than a fixed script. [orchestration_patterns.md#2]
- A fixed plan makes dependencies explicit and checkable. [orchestration_patterns.md#0]
## Limitations
- Only 2 documents discuss the blackboard pattern, so the comparison is thin.
## Sources
- orchestration_patterns.md
"""


def test_good_draft_passes():
    assert critique(GOOD) == {"score": 1.0, "verdict": "accept", "issues": []}


def test_flags_sources_listed_but_never_cited():
    result = critique(
        GOOD.replace("- orchestration_patterns.md", "- orchestration_patterns.md\n- memory_for_agents.md")
    )
    assert result["verdict"] == "revise"
    assert any("never cited: memory_for_agents.md" in i for i in result["issues"])


def test_flags_off_topic_findings():
    off_topic = GOOD.replace(
        "## Limitations",
        "- Shared memory between agents introduces risk. [memory_for_agents.md#3]\n"
        "- Episodic memory stores what happened. [memory_for_agents.md#1]\n"
        "- Tracing shows where time goes. [evaluation_and_operations.md#1]\n"
        "- Budgets prevent runaway loops. [evaluation_and_operations.md#3]\n## Limitations",
    ).replace(
        "- orchestration_patterns.md",
        "- orchestration_patterns.md\n- memory_for_agents.md\n- evaluation_and_operations.md",
    )
    result = critique(off_topic)
    assert any("do not address the question" in i for i in result["issues"])


def test_flags_generic_limitations():
    generic = GOOD.replace(
        "- Only 2 documents discuss the blackboard pattern, so the comparison is thin.",
        "- Claims were not independently verified.",
    )
    assert any("limitations are generic" in i.lower() for i in critique(generic)["issues"])


def test_rewording_without_fixing_does_not_pass_and_is_reported():
    """A writer that ignores feedback and only rewords must not get an accepting score."""
    calls = {"n": 0}

    def rewording_writer(messages, meta):
        calls["n"] += 1
        opener = ["In short,", "To summarise,", "Overall,"][calls["n"] % 3]
        return (
            f"# Answer\n{opener} blackboards are flexible.\n## Key findings\n"
            "- The blackboard pattern coordinates sources through a shared workspace. [orchestration_patterns.md#2]\n"
            "## Sources\n- orchestration_patterns.md\n- memory_for_agents.md\nCONFIDENCE: 0.9"
        )

    result = asyncio.run(run_pattern("hierarchical", QUESTION, llm=MockLLM({"writer": rewording_writer})))
    assert result.review.score < 0.8
    log = result.extra["revision_log"]
    assert log[-1]["fixed"] == [] and log[-1]["still_open"]
    assert result.revisions == 1  # stall detected after the first unproductive revision, no second loop
    assert result.extra["unresolved_issues"]


def test_sources_always_match_citations():
    for pattern in ("hierarchical", "blackboard", "sequential"):
        answer = asyncio.run(run_pattern(pattern, QUESTION)).answer
        cited = {s.split("#")[0] for s in re.findall(r"\[([^\]]*\.md[^\]]*)\]", answer)}
        listed = {ln[2:].strip() for ln in answer.split("## Sources")[1].splitlines() if ln.startswith("- ")}
        assert listed == cited, pattern


def test_off_topic_question_is_refused_not_padded():
    result = asyncio.run(run_pattern("hierarchical", "What is the best biryani in Hyderabad?"))
    assert "no evidence" in result.answer.lower()
    assert not re.search(r"\[[^\]]*\.md", result.answer)  # nothing cited that doesn't answer it
    assert set(result.extra["follow_ups"]) == {"t1", "t2"}  # researchers noticed and searched again


def test_blackboard_stops_when_revision_fixes_nothing():
    def stuck_writer(messages, meta):
        return "# A\n## Key findings\n- Blackboards coordinate sources. [orchestration_patterns.md#2]\nCONFIDENCE: 0.8"

    crew = build_research_crew(MockLLM({"writer": stuck_writer}))
    result = asyncio.run(BlackboardOrchestrator(crew).run(QUESTION))
    assert result.extra["schedule"].count("writer") == 2  # one draft, one revision, then it stops
    assert result.extra["unresolved_issues"]


def test_revision_log_classifies_issues():
    reviews = [
        Review(0.5, "revise", ["Add a '## Limitations' section.", "3 of 5 findings do not address the question."]),
        Review(0.75, "revise", ["1 of 4 findings do not address the question.", "Sources are listed but never cited."]),
    ]
    (entry,) = revision_log(reviews)
    assert entry["fixed"] == ["Add a '## Limitations' section."]
    assert entry["still_open"] == ["1 of 4 findings do not address the question."]  # counts ignored
    assert entry["new"] == ["Sources are listed but never cited."]
    assert not stalled([entry])


def test_bm25_prefers_rare_meaningful_terms():
    """Common words ("agents", "should") appear everywhere, so the rare one ("blackboard") decides."""
    sm = SemanticMemory()
    common = [
        "Agents should plan carefully and agents should check every plan that agents make.",
        "Agents should retry failed tool calls, and agents should log every failure.",
        "Agents should share memory carefully because agents can contaminate each other.",
        "Agents should be evaluated end to end, and agents should report their cost.",
    ]
    sm.add_document("\n\n".join(common + ["The blackboard pattern lets agents post to a shared workspace."]), "doc.md")
    top, _ = sm.search("When should agents use a blackboard?", k=1)[0]
    assert top.id == "doc.md#4"
