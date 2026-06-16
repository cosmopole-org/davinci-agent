"""Tests for the sandbox_pc tool + session-sandbox lifecycle wiring.

These tests cover:
1. The tool.py invoke surface (input validation + envelope shape sent to the
   sandbox creature over the bridge).
2. The correlation-id-matched reply flow (a stub bridge plays the role of the
   sandbox creature and signals a result back; the tool returns it).
3. The runtime's `_SessionSandbox` context manager: it must signal `start` on
   entry and `suspend` on exit when a sandbox_pc tool is configured, and be a
   silent no-op when one isn't.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_REPO = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO / "tools" / "sandbox_pc" / "tool.py"


def _load_tool_module(bridge_factory: Callable[[], Any]):
    """Load tools/sandbox_pc/tool.py with a stub tool_runtime.get_bridge()."""
    # Stub tool_runtime so the tool module's _get_bridge() returns our fake.
    stub = type(sys)("tool_runtime")
    stub.get_bridge = bridge_factory  # type: ignore[attr-defined]
    sys.modules["tool_runtime"] = stub
    spec = importlib.util.spec_from_file_location(
        "sandbox_pc_tool_" + str(id(bridge_factory)), _TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeBridge:
    """Records signal_user calls and lets the test replay an `on_signal` reply."""

    def __init__(self, *, machine_id: str = "davinci-machine", session_id: int = 7,
                 reply_with: Optional[Dict[str, Any]] = None) -> None:
        self.machine_id = machine_id
        self.creature_id = ""
        self.session_id = session_id
        self.sent: List[Dict[str, Any]] = []
        self._reply_with = reply_with
        self._handler: Optional[Callable[[Any], None]] = None

    def on_signal(self, handler: Callable[[Any], None]) -> None:
        self._handler = handler

    def signal_user(self, key: str, user_id: str, packet: Any) -> Dict[str, Any]:
        self.sent.append({"key": key, "user_id": user_id, "packet": packet})
        if self._reply_with is not None and self._handler is not None:
            # Replay the sandbox creature's signalUser("creatures/signal/result")
            # response on a background thread so the waiter sees ev.set().
            reply = dict(self._reply_with)
            reply["correlationId"] = packet["correlationId"]
            threading.Thread(target=self._handler, args=(reply,), daemon=True).start()
        return {"ok": True}


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #

def test_invoke_requires_command_for_exec():
    mod = _load_tool_module(lambda: None)
    out = mod.invoke("exec", {})
    assert out["ok"] is False
    assert "command" in out["error"]


def test_invoke_requires_content_for_write():
    mod = _load_tool_module(lambda: None)
    out = mod.invoke("write", {"file_name": "a.txt", "target_path": "/tmp"})
    assert out["ok"] is False
    assert "content" in out["error"]


def test_invoke_reports_no_bridge_for_status():
    mod = _load_tool_module(lambda: None)
    out = mod.invoke("status", {})
    assert out["ok"] is False
    assert "bridge" in out["error"]


# --------------------------------------------------------------------------- #
# End-to-end signaling with a stub bridge playing the sandbox creature
# --------------------------------------------------------------------------- #

def test_exec_signals_sandbox_and_returns_correlated_result():
    bridge = _FakeBridge(reply_with={"ok": True, "output": "hello\n", "vmId": "sess-davinci-7"})
    mod = _load_tool_module(lambda: bridge)

    out = mod.invoke("exec", {"command": "echo hello"})

    assert out["ok"] is True
    assert out["output"] == "hello\n"
    assert out["action"] == "exec"
    assert out["session_id"] == "davinci-7"  # derived from bridge.session_id

    # The signal goes on `creatures/signal` to the default sandbox target,
    # and the envelope carries the action, payload, correlation id, and reply_to.
    assert len(bridge.sent) == 1
    pkt = bridge.sent[0]
    assert pkt["key"] == "creatures/signal"
    assert pkt["user_id"] == "sandbox"
    env = pkt["packet"]
    assert env["function"] == "exec"
    assert env["action"] == "exec"
    assert env["payload"]["command"] == "echo hello"
    assert env["payload"]["sessionId"] == "davinci-7"
    assert env["reply_to"] == "davinci-machine"
    assert isinstance(env["correlationId"], str) and len(env["correlationId"]) > 0


def test_suspend_uses_short_timeout_and_resolves_session_id_from_payload():
    bridge = _FakeBridge(reply_with={"ok": True, "status": "suspended"})
    mod = _load_tool_module(lambda: bridge)

    out = mod.invoke("suspend", {"session_id": "abc-123"})
    assert out["ok"] is True
    assert out["action"] == "suspend"
    assert bridge.sent[0]["packet"]["payload"]["sessionId"] == "abc-123"
    # Session id appears in the result for caller-side correlation.
    assert out["session_id"] == "abc-123"


# --------------------------------------------------------------------------- #
# Runtime lifecycle: _SessionSandbox + _find_sandbox_tool_spec
# --------------------------------------------------------------------------- #

def test_find_sandbox_tool_spec_locates_by_tool_id():
    from davinci import caspar_runtime as rt
    spec = rt._find_sandbox_tool_spec({
        "caspar__web_search": {"tool_id": "web_search"},
        "caspar__sandbox_pc": {"tool_id": "sandbox_pc", "machine_id": "m1"},
    })
    assert spec is not None
    assert spec["tool_id"] == "sandbox_pc"
    assert spec["machine_id"] == "m1"


def test_find_sandbox_tool_spec_returns_none_when_absent():
    from davinci import caspar_runtime as rt
    assert rt._find_sandbox_tool_spec({"caspar__web_search": {"tool_id": "web_search"}}) is None


def test_session_sandbox_signals_start_and_suspend():
    from davinci import caspar_runtime as rt

    bridge = _FakeBridge()
    spec = {"tool_id": "sandbox_pc", "machine_id": "sandbox-machine-xyz"}
    with rt._SessionSandbox(bridge, spec, session_id="sid-1"):
        # Inside the loop: VM is being kept warm — start was signalled.
        assert len(bridge.sent) == 1
        first = bridge.sent[0]
        assert first["user_id"] == "sandbox-machine-xyz"
        assert first["packet"]["function"] == "start"
        assert first["packet"]["payload"]["sessionId"] == "sid-1"

    # On exit: VM is suspended (disk kept).
    assert len(bridge.sent) == 2
    second = bridge.sent[1]
    assert second["packet"]["function"] == "suspend"
    assert second["packet"]["payload"]["sessionId"] == "sid-1"


def test_session_sandbox_is_noop_without_spec():
    from davinci import caspar_runtime as rt
    bridge = _FakeBridge()
    with rt._SessionSandbox(bridge, None, session_id="x"):
        pass
    assert bridge.sent == []


def test_session_sandbox_is_noop_without_bridge():
    from davinci import caspar_runtime as rt
    spec = {"tool_id": "sandbox_pc", "machine_id": "m"}
    # Must not raise; just a silent no-op.
    with rt._SessionSandbox(None, spec, session_id="x"):
        pass
