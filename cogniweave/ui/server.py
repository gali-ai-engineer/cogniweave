"""Live web UI, built on the standard library only.

``python -m cogniweave ui`` starts a local server. The page asks for a run and
the server streams every trace span and blackboard post to the browser as
Server-Sent Events while the agents work. ``--export DIR`` records a set of
runs into ``replay.json`` next to a copy of the page, which can then be hosted
statically (e.g. GitHub Pages) and played back without a server.
"""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..cognition.memory import MemorySystem
from ..core.blackboard import Entry
from ..core.tracing import Span, Tracer
from ..llm.mock import MockLLM
from ..llm.providers import get_llm
from ..orchestration.result import RunResult
from ..runner import PATTERNS, build_orchestrator

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
DEFAULT_QUESTION = "What are the trade-offs between multi-agent and single-agent LLM systems?"
SPEEDS = {"watchable": 0.35, "fast": 0.12, "instant": 0.0}
Emit = Callable[[dict[str, Any]], None]


# ------------------------------------------------------------------ event encoding
def span_event(kind: str, span: Span) -> dict[str, Any]:
    return {
        "type": {"start": "span_start", "end": "span_end", "event": "event"}[kind],
        "t": round(span.end if kind == "end" and span.end is not None else span.start, 4),
        "id": span.id,
        "parent_id": span.parent_id,
        "name": span.name,
        "agent": span.agent,
        "start": round(span.start, 4),
        "end": None if span.end is None else round(span.end, 4),
        "status": span.status,
        "attrs": span.attrs,
    }


def entry_event(entry: Entry, t: float) -> dict[str, Any]:
    data = asdict(entry)
    data["tags"] = sorted(entry.tags)
    data["refs"] = list(entry.refs)
    return {"type": "board", "t": round(t, 4), "entry": data}


def memory_snapshot(memory: MemorySystem) -> dict[str, Any]:
    lessons = [
        {"agent": e.agent, "text": e.content, "task": e.task}
        for e in memory.episodic.episodes
        if e.kind == "reflection"
    ]
    strategies = [
        {"role": role, "strategy": s, "n": int(n), "mean": round(total / n, 3) if n else None}
        for role, table in memory.procedural.stats.items()
        for s, (n, total) in table.items()
    ]
    return {"episodes": len(memory.episodic.episodes), "lessons": lessons, "strategies": strategies}


def result_event(result: RunResult, memory: MemorySystem, elapsed: float, t: float) -> dict[str, Any]:
    keep = ("schedule", "follow_ups", "rounds", "revision_log", "unresolved_issues")
    return {
        "type": "result",
        "t": round(t, 4),
        "pattern": result.pattern,
        "answer": result.answer,
        "review": result.review.to_dict() if result.review else None,
        "plan": result.plan.to_dict() if result.plan else None,
        "revisions": result.revisions,
        "usage": dict(result.tracer.usage),
        "totals": {"llm_calls": result.tracer.total_calls, "tokens": result.tracer.total_tokens},
        "elapsed_s": round(elapsed, 2),
        "extra": {k: v for k, v in result.extra.items() if k in keep},
        "mermaid": result.tracer.to_mermaid(),
        "memory": memory_snapshot(memory),
    }


def _jsonable(event: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(event, default=str))


# ------------------------------------------------------------------ running
def execute_run(
    *,
    pattern: str,
    question: str,
    emit: Emit,
    memory: MemorySystem,
    provider: str = "mock",
    model: str | None = None,
    latency_s: float = SPEEDS["watchable"],
    corpus_dir: str | Path | None = None,
    label: str | None = None,
) -> RunResult | None:
    """Run one pattern synchronously, emitting UI events. Always ends with a ``done`` event."""
    tracer = Tracer(max_llm_calls=80)
    started = time.perf_counter()
    result = None
    try:
        if pattern not in PATTERNS:
            raise ValueError(f"Unknown pattern {pattern!r}")
        llm = MockLLM(latency_s=latency_s) if provider == "mock" else get_llm(provider, model)
        crew, orchestrator = build_orchestrator(pattern, llm=llm, memory=memory, tracer=tracer, corpus_dir=corpus_dir)
        tracer.subscribe(lambda kind, span: emit(_jsonable(span_event(kind, span))))
        crew.blackboard.subscribe("*", lambda entry: emit(_jsonable(entry_event(entry, tracer.now()))))
        emit(
            {
                "type": "run_start",
                "t": 0.0,
                "pattern": pattern,
                "question": question,
                "provider": provider,
                "label": label,
                "agents": [{"name": a.name, "role": a.role, "persona": a.persona} for a in crew.agents.values()],
                "memory": memory_snapshot(memory),
            }
        )
        result = asyncio.run(orchestrator.run(question))
        emit(_jsonable(result_event(result, memory, time.perf_counter() - started, tracer.now())))
    except Exception as exc:
        emit({"type": "error", "t": round(tracer.now(), 4), "message": f"{type(exc).__name__}: {exc}"})
    finally:
        emit({"type": "done", "t": round(tracer.now(), 4)})
    return result


class UIState:
    def __init__(self, memory_path: Path | None, corpus_dir: str | Path | None = None) -> None:
        self.memory_path = memory_path
        self.corpus_dir = corpus_dir
        self.busy = threading.Lock()
        self.memory = MemorySystem()
        if memory_path:
            self.memory.load(memory_path)

    def save(self) -> None:
        if self.memory_path:
            self.memory.save(self.memory_path)

    def reset(self) -> None:
        self.memory = MemorySystem()
        if self.memory_path and self.memory_path.exists():
            self.memory_path.unlink()


# ------------------------------------------------------------------ HTTP
def make_handler(state: UIState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "cogniweave-ui"

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the console quiet
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload, default=str).encode(), "application/json")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/health":
                self._json({"ok": True, "mode": "live", "patterns": list(PATTERNS), "speeds": list(SPEEDS)})
            elif url.path == "/api/memory":
                self._json(memory_snapshot(state.memory))
            elif url.path == "/api/run":
                self._stream_run(parse_qs(url.query))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/memory/reset":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            elif not state.busy.acquire(blocking=False):
                self._json({"error": "A run is in progress. Reset memory after it finishes."}, HTTPStatus.CONFLICT)
            else:
                try:
                    state.reset()
                finally:
                    state.busy.release()
                self._json(memory_snapshot(state.memory))

        def _stream_run(self, params: dict[str, list[str]]) -> None:
            def arg(name: str, default: str) -> str:
                return (params.get(name) or [default])[0].strip() or default

            if not state.busy.acquire(blocking=False):
                self._json({"error": "Another run is in progress."}, HTTPStatus.CONFLICT)
                return
            events: queue.Queue[dict[str, Any]] = queue.Queue()

            def worker() -> None:
                try:
                    execute_run(
                        pattern=arg("pattern", "hierarchical"),
                        question=arg("question", DEFAULT_QUESTION)[:500],
                        provider=arg("provider", "mock"),
                        model=(params.get("model") or [None])[0] or None,
                        latency_s=SPEEDS.get(arg("speed", "watchable"), SPEEDS["watchable"]),
                        memory=state.memory,
                        corpus_dir=state.corpus_dir,
                        emit=events.put,
                    )
                    state.save()
                finally:
                    state.busy.release()

            threading.Thread(target=worker, daemon=True).start()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while True:
                    try:
                        event = events.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(f"data: {json.dumps(event, default=str)}\n\n".encode())
                    self.wfile.flush()
                    if event["type"] == "done":
                        break
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser closed the tab; the run finishes in the background

    return Handler


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    memory_path: Path | None = Path(".cogniweave/memory.json"),
    corpus_dir: str | Path | None = None,
    open_browser: bool = True,
) -> None:
    state = UIState(memory_path, corpus_dir)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"cogniweave UI running at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


# ------------------------------------------------------------------ static replay
DEMO_RUNS = (
    ("hierarchical", "Planner and parallel researchers, run 1", True),
    ("hierarchical", "Same question, run 2: the writer remembers the lesson", False),
    ("blackboard", "Blackboard: no plan, agents bid for the next move", True),
    ("debate", "Debate: three viewpoints and a judge", True),
    ("sequential", "Sequential pipeline: the baseline", True),
)


def export_demo(out_dir: str | Path, *, question: str = DEFAULT_QUESTION, latency_s: float = 0.25) -> Path:
    """Record the demo runs to ``replay.json`` and copy the page beside it."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs, memory = [], MemorySystem()
    for i, (pattern, label, fresh_memory) in enumerate(DEMO_RUNS):
        if fresh_memory:
            memory = MemorySystem()
        events: list[dict[str, Any]] = []
        execute_run(
            pattern=pattern, question=question, emit=events.append, memory=memory, latency_s=latency_s, label=label
        )
        runs.append({"id": f"run-{i + 1}", "label": label, "pattern": pattern, "events": events})
    replay = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "runs": runs}
    (out / "replay.json").write_text(json.dumps(replay, default=str), encoding="utf-8")
    shutil.copyfile(INDEX_HTML, out / "index.html")
    return out
