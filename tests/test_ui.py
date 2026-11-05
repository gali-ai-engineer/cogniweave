import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from cogniweave.ui.server import INDEX_HTML, UIState, export_demo, make_handler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(UIState(memory_path=None)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read().decode()


def _stream(url):
    events = []
    with urllib.request.urlopen(url, timeout=30) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
                if events[-1]["type"] == "done":
                    break
    return events


def test_serves_page_and_health(server):
    status, body = _get(server + "/")
    assert status == 200 and "<title>cogniweave</title>" in body
    assert json.loads(_get(server + "/api/health")[1])["mode"] == "live"


def test_run_streams_events_and_memory_persists_between_runs(server):
    url = server + "/api/run?pattern=hierarchical&speed=instant"
    first = _stream(url)
    kinds = [e["type"] for e in first]
    assert kinds[0] == "run_start" and kinds[-1] == "done" and "result" in kinds
    assert {"span_start", "span_end", "board"} <= set(kinds)
    result = next(e for e in first if e["type"] == "result")
    assert result["revisions"] == 1 and result["memory"]["lessons"]
    (revision,) = result["extra"]["revision_log"]  # the page shows fixed / still open / new per revision
    assert revision["fixed"] and not revision["still_open"]
    assert result["extra"]["unresolved_issues"] == []

    second = next(e for e in _stream(url) if e["type"] == "result")
    assert second["revisions"] == 0  # the lesson from run 1 was recalled

    req = urllib.request.Request(server + "/api/memory/reset", method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        assert json.loads(r.read())["lessons"] == []


def test_bad_pattern_reports_error_and_finishes(server):
    events = _stream(server + "/api/run?pattern=telepathy&speed=instant")
    assert [e["type"] for e in events] == ["error", "done"]


def test_export_demo_writes_replay(tmp_path):
    out = export_demo(tmp_path, latency_s=0)
    replay = json.loads((out / "replay.json").read_text())
    assert [r["pattern"] for r in replay["runs"]] == [
        "hierarchical",
        "hierarchical",
        "blackboard",
        "debate",
        "sequential",
    ]
    results = [next(e for e in r["events"] if e["type"] == "result") for r in replay["runs"]]
    assert results[0]["revisions"] == 1 and results[1]["revisions"] == 0
    assert (out / "index.html").read_text() == INDEX_HTML.read_text()


def test_published_demo_page_is_up_to_date():
    published = ROOT / "docs" / "demo" / "index.html"
    if not published.exists():
        pytest.skip("docs/demo not generated")
    assert published.read_text() == INDEX_HTML.read_text(), "run: python -m cogniweave ui --export docs/demo"
