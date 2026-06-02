"""Tests for the tool-creature runtime and the offline creature entrypoint."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from contextlib import redirect_stdout

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Load the tool runtime module directly (it lives under tools/_runtime).
_spec = importlib.util.spec_from_file_location(
    "tool_runtime", os.path.join(REPO, "tools", "_runtime", "tool_runtime.py"))
tool_runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool_runtime)  # type: ignore


def test_python_exec_handler_runs_code():
    out = tool_runtime._dispatch("python_exec", "invoke", {"code": "result = 6 * 7"})
    assert out["ok"] is True and out["result"] == 42


def test_vector_search_handler_ranks():
    out = tool_runtime._dispatch("vector_search", "invoke", {"query": "caspar creatures"})
    assert out["ok"] is True and out["top"]


def test_unknown_tool_falls_back_to_echo():
    out = tool_runtime._dispatch("mystery_tool", "invoke", {"x": 1})
    assert out["ok"] is True and out["echo"] == {"x": 1}


def test_tool_runtime_main_emits_response(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "task.json").write_text(json.dumps(
        {"tool_id": "python_exec", "function": "invoke", "payload": {"code": "result = 1+1"}}))
    old = os.environ.get("DAVINCI_INPUT_DIR")
    os.environ["DAVINCI_INPUT_DIR"] = str(input_dir)
    tool_runtime.INPUT_DIR = str(input_dir)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = tool_runtime.main()
        out = buf.getvalue()
        assert rc == 0
        assert "TOOL_RESPONSE" in out
        line = [l for l in out.splitlines() if l.startswith("TOOL_RESPONSE")][0]
        resp = json.loads(line.split("TOOL_RESPONSE", 1)[1].strip())
        assert resp["result"]["result"] == 2
    finally:
        if old is None:
            os.environ.pop("DAVINCI_INPUT_DIR", None)
        else:
            os.environ["DAVINCI_INPUT_DIR"] = old


def test_offline_creature_entrypoint(tmp_path):
    from davinci import caspar_runtime
    os.environ["DAVINCI_TASK"] = "offline self-test"
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = caspar_runtime.main([])
        out = buf.getvalue()
        assert "DAVINCI_BOOT" in out and "DAVINCI_RESULT" in out
        assert rc in (0, 2)
    finally:
        os.environ.pop("DAVINCI_TASK", None)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
