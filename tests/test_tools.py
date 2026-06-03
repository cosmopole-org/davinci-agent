"""Tests for the real tool-creature implementations under tools/.

Each tool ships its own ``tool.py`` with an ``invoke()`` entrypoint. These
tests exercise the offline-safe behaviour of every tool (no network or external
credentials required) and skip cleanly when a tool's optional third-party
dependency isn't installed in the test environment.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(tool_id: str):
    path = os.path.join(REPO, "tools", tool_id, "tool.py")
    spec = importlib.util.spec_from_file_location(f"tool_{tool_id}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# --------------------------------------------------------------------------- #
# python_exec (stdlib only — always testable)
# --------------------------------------------------------------------------- #

def test_python_exec_runs_and_captures():
    tool = _load("python_exec")
    out = tool.invoke("execute", {"code": "print('hi')\nresult = sum(range(10))"})
    assert out["ok"] is True
    assert out["result"] == 45
    assert "hi" in out["stdout"]


def test_python_exec_timeout():
    tool = _load("python_exec")
    out = tool.invoke("execute", {"code": "while True:\n    pass", "timeout": 2})
    assert out["ok"] is False and out.get("timed_out") is True


def test_python_exec_reports_error():
    tool = _load("python_exec")
    out = tool.invoke("execute", {"code": "raise ValueError('boom')"})
    assert out["ok"] is False and "ValueError" in out["stderr"]


# --------------------------------------------------------------------------- #
# sql_query (needs SQLAlchemy)
# --------------------------------------------------------------------------- #

def test_sql_query_in_memory():
    pytest.importorskip("sqlalchemy")
    tool = _load("sql_query")
    out = tool.invoke("query", {"database_url": "sqlite:///:memory:", "sql": [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)",
        "INSERT INTO t (name) VALUES ('davinci'), ('caspar')",
        "SELECT name FROM t ORDER BY id",
    ]})
    assert out["ok"] is True
    rows = out["results"][-1]["rows"]
    assert [r["name"] for r in rows] == ["davinci", "caspar"]


def test_sql_query_read_only_rejects_writes():
    pytest.importorskip("sqlalchemy")
    tool = _load("sql_query")
    out = tool.invoke("query", {"database_url": "sqlite:///:memory:",
                                "sql": "DELETE FROM t", "read_only": True})
    assert out["ok"] is False and "read_only" in out["error"]


# --------------------------------------------------------------------------- #
# vector_search (needs numpy + scikit-learn for the hashing backend)
# --------------------------------------------------------------------------- #

def test_vector_search_one_shot():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    tool = _load("vector_search")
    docs = ["the cat sat on the mat", "caspar runs creatures across vm runtimes",
            "tool creatures respond to signals"]
    out = tool.invoke("vector_search", {"query": "how do creatures signal",
                                        "documents": docs, "top_k": 2, "backend": "hashing"})
    assert out["ok"] is True and len(out["results"]) == 2
    assert all("score" in r for r in out["results"])


def test_vector_search_persistent_index(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    tool = _load("vector_search")
    tool.DATA_DIR = str(tmp_path)
    col = "unit"
    idx = tool.invoke("vector_search", {"action": "index", "collection": col, "backend": "hashing",
                                        "documents": [{"id": "a", "text": "alpha beta gamma"},
                                                      {"id": "b", "text": "delta epsilon"}]})
    assert idx["ok"] and idx["indexed"] == 2
    hit = tool.invoke("vector_search", {"collection": col, "query": "alpha", "top_k": 1,
                                        "backend": "hashing"})
    assert hit["ok"] and hit["results"][0]["id"] == "a"


# --------------------------------------------------------------------------- #
# calendar_connector (ICS backend needs icalendar)
# --------------------------------------------------------------------------- #

def test_calendar_generate_ics():
    pytest.importorskip("icalendar")
    tool = _load("calendar_connector")
    out = tool.invoke("calendar_action", {"operation": "generate_ics", "provider": "ics",
                                          "summary": "Sync", "start": "2026-06-10T15:00:00",
                                          "end": "2026-06-10T15:30:00"})
    assert out["ok"] is True
    assert "BEGIN:VEVENT" in out["result"]["ics"]
    assert "SUMMARY:Sync" in out["result"]["ics"]


# --------------------------------------------------------------------------- #
# git_tool (needs the git binary)
# --------------------------------------------------------------------------- #

def test_git_status_commit(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git binary not available")
    tool = _load("git_tool")
    repo = str(tmp_path / "repo")
    os.makedirs(repo)
    with open(os.path.join(repo, "f.txt"), "w") as fh:
        fh.write("hello\n")
    out = tool.invoke("status_commit", {"repo_path": repo, "message": "init",
                                        "author_name": "T", "author_email": "t@e.co"})
    assert out["ok"] is True
    assert "init" in out["head"]


# --------------------------------------------------------------------------- #
# Connectors: real call path, clean credential errors without secrets.
# --------------------------------------------------------------------------- #

def test_slack_requires_token(monkeypatch):
    pytest.importorskip("slack_sdk")
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_TOKEN", raising=False)
    tool = _load("slack_connector")
    out = tool.invoke("slack_action", {"operation": "auth_test"})
    assert out["ok"] is False and "token" in out["error"].lower()


def test_jira_requires_config(monkeypatch):
    for var in ("JIRA_URL", "JIRA_EMAIL", "JIRA_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    tool = _load("jira_connector")
    out = tool.invoke("jira_action", {"operation": "myself"})
    assert out["ok"] is False and "Jira" in out["error"]


def test_web_search_requires_query():
    tool = _load("web_search")
    out = tool.invoke("search", {})
    assert out["ok"] is False and "query" in out["error"]


def test_fetch_url_requires_url():
    pytest.importorskip("requests")
    tool = _load("fetch_url")
    out = tool.invoke("fetch", {})
    assert out["ok"] is False and "url" in out["error"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
