"""Deterministic offline LLM.

``MockLLM`` lets the whole system — planning, tool use, critique, reflection,
debate — run in CI and demos with no API key. Each role has a small rule-based
handler that reads the structured prompt (``<task>``, ``<context>`` ...) the
agents send. It is *not* trying to be smart; it is trying to be faithful to the
protocols real models are asked to follow, so the orchestration code paths are
exercised exactly as they would be with a hosted model.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from ..cognition.embeddings import tokenize
from .base import LLMResponse, Message, estimate_tokens

Handler = Callable[[list[Message], dict[str, Any]], str]

CITED_BULLET = re.compile(r"^\s*-\s+(?P<claim>.+?)\s*\[(?P<src>[^\]]+)\]\s*$", re.M)
OBS_LINE = re.compile(r"^\[(?P<src>[^\]]+)\]\s+(?P<text>.+)$", re.M)
TRADEOFF_WORDS = (
    "cost",
    "risk",
    "overhead",
    "latency",
    "fail",
    "however",
    "but ",
    "trade",
    "error",
    "complex",
    "harder",
    "expensive",
    "drift",
    "loop",
)


def tag(text: str, name: str) -> str:
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else ""


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]


def _dedupe(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out = []
    for claim, src in pairs:
        key = claim.lower().strip(" .")
        if key not in seen:
            seen.add(key)
            out.append((claim, src))
    return out


def _is_tradeoff(claim: str) -> bool:
    low = claim.lower()
    return any(w in low for w in TRADEOFF_WORDS)


# Words too common in this domain to say whether a claim is on topic.
GENERIC_TERMS = frozenset(
    tokenize(
        "agent agents ai llm llms model models system systems language better best good should use using "
        "make making different between versus way ways work works thing things"
    )
)


def key_terms(goal: str) -> set[str]:
    """The words in the question that actually distinguish it."""
    return set(tokenize(goal)) - GENERIC_TERMS


def named_terms(goal: str, limit: int = 4) -> str:
    """The question's distinctive words as the user wrote them (not their internal stems)."""
    terms, words = key_terms(goal), []
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", goal):
        stems = tokenize(word)
        if stems and stems[0] in terms and word.lower() not in words:
            words.append(word.lower())
    return _join(words[:limit])


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " or " + items[-1]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def relevance(text: str, terms: set[str]) -> float:
    """Share of the question's key terms that a claim mentions (1.0 when there are none to check)."""
    if not terms:
        return 1.0
    return len(terms & set(tokenize(text))) / len(terms)


def _rank(claims: list[tuple[str, str]], terms: set[str]) -> list[tuple[str, str]]:
    """Most relevant first; retrieval order breaks ties."""
    return sorted(claims, key=lambda c: -relevance(c[0], terms))


def _section(markdown: str, heading: str) -> str:
    m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", markdown, re.M | re.S)
    return m.group(1) if m else ""


def _bullets(text: str) -> list[str]:
    return [ln.strip()[2:].strip() for ln in text.splitlines() if ln.strip().startswith("- ")]


# --------------------------------------------------------------------- handlers
def _planner(messages: list[Message], meta: dict[str, Any]) -> str:
    prompt = messages[0].content
    goal = tag(prompt, "goal") or tag(prompt, "task")
    if meta.get("phase") == "questions":
        return json.dumps(
            {
                "questions": [
                    f"What are the core concepts and benefits behind: {goal}",
                    f"What are the costs, risks and failure modes of: {goal}",
                ]
            }
        )
    plan = {
        "goal": goal,
        "tasks": [
            {
                "id": "t1",
                "role": "researcher",
                "depends_on": [],
                "description": f"Research the core concepts and benefits relevant to: {goal}",
            },
            {
                "id": "t2",
                "role": "researcher",
                "depends_on": [],
                "description": f"Research the costs, risks and failure modes relevant to: {goal}",
            },
            {
                "id": "t3",
                "role": "analyst",
                "depends_on": ["t1", "t2"],
                "description": "Synthesise the research into key findings and trade-offs, keeping citations",
            },
            {
                "id": "t4",
                "role": "writer",
                "depends_on": ["t3"],
                "description": "Write a concise, cited markdown report answering the goal",
            },
        ],
    }
    return "Plan:\n```json\n" + json.dumps(plan, indent=2) + "\n```"


def _researcher(messages: list[Message], meta: dict[str, Any]) -> str:
    prompt = messages[0].content
    task, goal = tag(prompt, "task"), tag(prompt, "goal")
    observations = [m.content for m in messages if m.role == "user" and m.content.startswith("OBSERVATION:")]
    if not observations:
        # Search for the question itself; the task wording only decides the angle.
        angle = " risks costs limitations" if any(w in task.lower() for w in ("risk", "cost", "fail")) else ""
        terms = key_terms(goal)
        # A second, deeper attempt searches only the question's distinctive terms.
        focused = task.startswith("Deepen") and terms
        query = " ".join(sorted(terms)) if focused else ((goal + angle) if goal else task)
        args = json.dumps({"query": query, "k": 4})
        return f"THOUGHT: I need evidence from the knowledge base before answering.\nACTION: corpus_search {args}"
    hits = OBS_LINE.findall("\n".join(observations))
    bullets, seen = [], set()
    for src, text in hits:
        for s in sentences(text)[:2]:
            if s not in seen:
                seen.add(s)
                bullets.append(f"- {s} [{src}]")
    # Metacognition: confident only if the evidence actually mentions what was asked.
    on_topic = [h for h in hits if relevance(h[1], key_terms(goal)) > 0]
    confidence = 0.85 if len(hits) >= 2 and on_topic else 0.35
    body = "\n".join(bullets) or "- No relevant evidence was found in the knowledge base. [none]"
    return f"FINDINGS:\n{body}\nCONFIDENCE: {confidence}"


def _analyst(messages: list[Message], meta: dict[str, Any]) -> str:
    prompt = messages[0].content
    claims = _dedupe(CITED_BULLET.findall(tag(prompt, "context")))
    claims = _rank([c for c in claims if c[1] != "none"], key_terms(tag(prompt, "goal")))
    findings = [c for c in claims if not _is_tradeoff(c[0])][:6]
    tradeoffs = [c for c in claims if _is_tradeoff(c[0])][:6]
    lines = ["## Analysis", "### Key findings"]
    lines += [f"- {c} [{s}]" for c, s in findings] or ["- (none)"]
    lines += ["### Trade-offs and risks"]
    lines += [f"- {c} [{s}]" for c, s in tradeoffs] or ["- (none)"]
    confidence = 0.8 if findings and tradeoffs else 0.5
    return "\n".join(lines) + f"\nCONFIDENCE: {confidence}"


def _writer(messages: list[Message], meta: dict[str, Any]) -> str:
    prompt = messages[0].content
    goal = tag(prompt, "goal") or tag(prompt, "task")
    guidance = (tag(prompt, "feedback") + " " + tag(prompt, "lessons")).lower()
    terms = key_terms(goal)
    claims = _rank([c for c in _dedupe(CITED_BULLET.findall(tag(prompt, "context"))) if c[1] != "none"], terms)
    if "do not address the question" in guidance:  # critic flagged off-topic findings: drop them, don't reword
        claims = [c for c in claims if relevance(c[0], terms) > 0]
    findings = [c for c in claims if not _is_tradeoff(c[0])][:5]
    tradeoffs = [c for c in claims if _is_tradeoff(c[0])][:5]
    shown = findings + tradeoffs
    out = [f"# {goal.rstrip('?')}", "", "## Executive summary"]
    summary = [f"{c} [{s}]" for c, s in (findings[:1] + tradeoffs[:1])]
    out.append(
        " ".join(summary)
        or "The knowledge base has no evidence that directly answers this question, so no answer is given."
    )
    out += ["", "## Key findings", *([f"- {c} [{s}]" for c, s in findings] or ["- No direct evidence found."])]
    out += ["", "## Trade-offs", *([f"- {c} [{s}]" for c, s in tradeoffs] or ["- No direct evidence found."])]
    docs = sorted({s.split("#")[0] for _, s in shown})
    if "limitation" in guidance:
        out += ["", "## Limitations", *_specific_limitations(shown, docs, terms, named_terms(goal))]
    out += ["", "## Sources", *[f"- {d}" for d in docs]]  # only documents actually cited above
    return "\n".join(out) + "\nCONFIDENCE: 0.8"


def _specific_limitations(shown: list[tuple[str, str]], docs: list[str], terms: set[str], named: str) -> list[str]:
    """Limitations about *this* answer's evidence, not boilerplate."""
    lines = []
    if terms:
        on_topic = sum(1 for c, _ in shown if relevance(c, terms) > 0)
        if not shown:
            lines.append(f"- No document in the knowledge base mentions {named}.")
        elif len(shown) == 1:
            lines.append(f"- The answer rests on a single cited finding about {named}, which is thin evidence.")
        elif on_topic == len(shown):
            lines.append(f"- All {len(shown)} cited findings mention {named}, but they come from a small corpus.")
        else:
            lines.append(
                f"- Only {on_topic} of the {len(shown)} cited findings directly mention {named}; "
                "the rest is supporting context."
            )
    counts = Counter(s.split("#")[0] for _, s in shown)
    if len(shown) > 1:
        top, n = counts.most_common(1)[0]
        if n > len(shown) / 2:
            share = f"All {n}" if n == len(shown) else f"{n} of the {len(shown)}"
            lines.append(f"- {share} findings come from {top} alone, so they are not cross-checked.")
    if docs:
        lines.append(
            f"- All evidence comes from {_plural(len(docs), 'local document')} ({', '.join(docs)}); "
            "nothing newer or external was checked."
        )
    return lines


def _critic(messages: list[Message], meta: dict[str, Any]) -> str:
    """Checks substance as well as structure, so a reworded draft cannot pass."""
    prompt = messages[0].content
    draft, terms = tag(prompt, "context"), key_terms(tag(prompt, "goal"))
    issues = []
    cited = {src.split("#")[0] for src in re.findall(r"\[([^\]]*\.md[^\]]*)\]", draft)}

    # 1. structure and grounding
    if "## Sources" not in draft:
        issues.append("Add a '## Sources' section listing every cited document.")
    if len(re.findall(r"\[[^\]]*\.md[^\]]*\]", draft)) < 3:
        issues.append("Ground more claims with inline citations (at least three).")

    # 2. every listed source must actually be cited
    uncited = sorted(set(_bullets(_section(draft, "Sources"))) - cited)
    if uncited:
        issues.append(f"Sources are listed but never cited: {', '.join(uncited)}. Cite them or remove them.")

    # 3. the findings must answer *this* question
    claims = CITED_BULLET.findall(draft)
    off_topic = [c for c, _ in claims if relevance(c, terms) == 0]
    if terms and claims and len(off_topic) / len(claims) > 0.4:
        issues.append(
            f"{len(off_topic)} of {len(claims)} findings do not address the question "
            f"(none mention {named_terms(tag(prompt, 'goal'), limit=6)}). Drop them rather than rewording them."
        )

    # 4. limitations must be about this answer, not boilerplate
    limitations = _section(draft, "Limitations")
    if "## Limitations" not in draft:
        issues.append("Add a '## Limitations' section that states the specific evidence gaps for this question.")
    elif not any(re.search(r"\d|\.md", line) or relevance(line, terms) > 0 for line in _bullets(limitations)):
        issues.append("The limitations are generic. Name the specific evidence gaps for this question.")

    score = round(max(0.0, 1.0 - 0.25 * len(issues)), 2)
    return json.dumps({"score": score, "verdict": "accept" if not issues else "revise", "issues": issues})


def _reflector(messages: list[Message], meta: dict[str, Any]) -> str:
    feedback = tag(messages[0].content, "feedback")
    try:
        issues = json.loads(feedback).get("issues", [])
    except (json.JSONDecodeError, AttributeError):
        issues = [feedback] if feedback else []
    if not issues:
        return "LESSON: The approach worked; keep the same structure next time."
    return "LESSON: Reviewers expect this up front - " + " ".join(issues)


_PERSONA_KEYS = {
    "advocate": ("improve", "special", "parallel", "better", "outperform", "divide", "focus"),
    "skeptic": TRADEOFF_WORDS,
    "pragmatist": ("when", "depend", "only", "simple", "start", "should"),
}
_PERSONA_POSITION = {
    "advocate": "Specialised, coordinated agents are the stronger design for this problem.",
    "skeptic": "The coordination cost usually outweighs the benefit; a single well-tooled agent is preferable.",
    "pragmatist": "It depends on how decomposable the task is: start with one agent and add agents only where "
    "specialisation or parallelism clearly pays off.",
}


def _debater(messages: list[Message], meta: dict[str, Any]) -> str:
    prompt = messages[0].content
    persona = meta.get("persona", "pragmatist")
    keys = _PERSONA_KEYS.get(persona, _PERSONA_KEYS["pragmatist"])
    evidence = OBS_LINE.findall(tag(prompt, "context"))
    picks = [(s, src) for src, text in evidence for s in sentences(text) if any(k in s.lower() for k in keys)]
    picks = _dedupe(picks)[:2] or [(sentences(t)[0], s) for s, t in evidence[:2] if sentences(t)]
    lines = [f"POSITION: {_PERSONA_POSITION.get(persona, _PERSONA_POSITION['pragmatist'])}", "EVIDENCE:"]
    lines += [f"- {s} [{src}]" for s, src in picks]
    if int(meta.get("round", 1)) > 1:
        peers = re.findall(r"^### (\S+)\nPOSITION: (.+)$", tag(prompt, "feedback"), re.M)
        others = [name for name, _ in peers if name != meta.get("agent")]
        if others:
            lines.append(
                f"REBUTTAL: Having read {', '.join(others)}, I keep my core position but accept that the answer "
                "is conditional on task structure and measured evaluation."
            )
    return "\n".join(lines) + "\nCONFIDENCE: 0.7"


def _judge(messages: list[Message], meta: dict[str, Any]) -> str:
    ctx = tag(messages[0].content, "context")
    positions = re.findall(r"^### (\S+)\nPOSITION: (.+)$", ctx, re.M)
    claims = _dedupe(CITED_BULLET.findall(ctx))
    counts = Counter(src.split("#")[0] for _, src in claims)
    chosen = next((p for n, p in positions if "pragmatist" in n), positions[0][1] if positions else "No verdict.")
    out = ["## Verdict", f"Final answer: {chosen}", "", "### Positions considered"]
    out += [f"- **{n}**: {p}" for n, p in positions]
    out += ["", "### Strongest evidence", *[f"- {c} [{s}]" for c, s in claims[:4]]]
    out += ["", "### Most-cited sources", *[f"- {s} ({n} citations)" for s, n in counts.most_common(3)]]
    return "\n".join(out) + "\nCONFIDENCE: 0.8"


def _generic(messages: list[Message], meta: dict[str, Any]) -> str:
    return f"Completed: {tag(messages[0].content, 'task') or messages[-1].content[:80]}\nCONFIDENCE: 0.6"


DEFAULT_HANDLERS: dict[str, Handler] = {
    "planner": _planner,
    "researcher": _researcher,
    "analyst": _analyst,
    "writer": _writer,
    "critic": _critic,
    "reflector": _reflector,
    "debater": _debater,
    "judge": _judge,
}


class MockLLM:
    """Rule-based LLM keyed on ``metadata['role']``. Override any role via ``handlers``."""

    name = "mock"

    def __init__(self, handlers: dict[str, Handler] | None = None, latency_s: float = 0.0) -> None:
        self.handlers = {**DEFAULT_HANDLERS, **(handlers or {})}
        self.latency_s = latency_s  # simulate network latency to exercise concurrency
        self.calls: list[str] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        meta = metadata or {}
        role = meta.get("role", "generic")
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        text = self.handlers.get(role, _generic)(messages, meta)
        self.calls.append(role)
        prompt = system + "".join(m.content for m in messages)
        return LLMResponse(text, "mock-1", estimate_tokens(prompt), estimate_tokens(text))
