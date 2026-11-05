from cogniweave.cognition.embeddings import HashingEmbedder, cosine
from cogniweave.cognition.memory import (
    Episode,
    EpisodicMemory,
    MemorySystem,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)


def test_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.embed("multi agent orchestration"), e.embed("multi agent orchestration")
    assert a == b
    assert abs(cosine(a, a) - 1.0) < 1e-9
    assert cosine(a, e.embed("multi agent systems")) > cosine(a, e.embed("banana bread recipe"))


def test_working_memory_evicts_least_salient_under_budget():
    wm = WorkingMemory(budget_tokens=60)
    wm.add("x" * 120, salience=1.0, kind="context")  # ~30 tokens
    wm.add("y" * 120, salience=0.2, kind="experience")
    wm.add("z" * 120, salience=0.9, kind="lesson")
    assert wm.used_tokens <= 60
    assert [i.kind for i in wm.evicted] == ["experience"]
    assert wm.render("lesson").startswith("z")


def test_working_memory_truncates_single_oversized_item():
    wm = WorkingMemory(budget_tokens=10)
    wm.add("a" * 1000, salience=1.0, kind="context")
    assert "[truncated]" in wm.render("context")


def test_episodic_recall_balances_relevance_recency_importance():
    now = [1_000_000.0]
    mem = EpisodicMemory(clock=lambda: now[0], half_life_s=100)
    old_important = mem.add(Episode("w", "write report", "include limitations", "reflection", 0.9, ts=0))
    fresh_trivial = mem.add(Episode("w", "write report", "drafted report", "action", 0.1, ts=now[0]))
    mem.add(Episode("w", "bake bread", "used yeast", "action", 0.9, ts=now[0]))
    results = mem.recall("write the report", k=3)
    ids = [e.id for e, _ in results]
    assert old_important.id in ids and fresh_trivial.id in ids
    assert all("bread" not in e.task for e, _ in results)  # irrelevant memory filtered
    assert mem.recall("write report", agent="nobody") == []


def test_semantic_memory_chunks_and_searches():
    sm = SemanticMemory()
    n = sm.add_document(
        "# Title\n\nAgents coordinate through a shared blackboard workspace.\n\n"
        "Bananas are a good source of potassium for athletes.",
        "doc.md",
    )
    assert n == 2
    top, score = sm.search("blackboard coordination", k=1)[0]
    assert top.id == "doc.md#0" and score > 0


def test_procedural_memory_ucb_explores_then_exploits():
    pm = ProceduralMemory(exploration=0.1)
    strategies = ["a", "b"]
    assert pm.select("writer", strategies) == "a"
    pm.update("writer", "a", 0.2)
    assert pm.select("writer", strategies) == "b"  # untried strategy is explored
    pm.update("writer", "b", 0.9)
    for _ in range(5):
        choice = pm.select("writer", strategies)
        pm.update("writer", choice, 0.9 if choice == "b" else 0.2)
    assert pm.select("writer", strategies) == "b"
    assert pm.mean("writer", "b") > pm.mean("writer", "a")


def test_memory_system_round_trips(tmp_path):
    m = MemorySystem()
    m.episodic.add(Episode("writer", "task", "lesson text", "reflection", 0.9))
    m.procedural.update("writer", "s1", 0.7)
    path = tmp_path / "mem.json"
    m.save(path)
    m2 = MemorySystem()
    assert m2.load(path)
    assert m2.episodic.episodes[0].content == "lesson text"
    assert m2.procedural.mean("writer", "s1") == 0.7
    assert m2.episodic.recall("task")
