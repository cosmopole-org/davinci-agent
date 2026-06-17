"""Tests for the sandbox_pc tool + SandboxManager + session lifecycle.

The sandbox_pc tool now manages per-session VMs *directly* through the vmm host
functions over the bridge (runVm / execVm / terminateVm / copyToVm) — there is
no intermediary creature. These tests drive the real tool.py and the real
davinci.sandbox_core.SandboxManager against a fake bridge that records the host
calls and returns host-shaped results.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_REPO = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO / "tools" / "sandbox_pc" / "tool.py"


class _FakeBridge:
    """Records bridge.call(op, input) and returns canned host results."""

    def __init__(self, *, session_id: int = 7, machine_id: str = "m-davinci",
                 responses: Optional[Dict[str, Any]] = None) -> None:
        self.session_id = session_id
        self.machine_id = machine_id
        self.calls: List[Dict[str, Any]] = []
        self.responses = responses or {}

    def call(self, op: str, input: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append({"op": op, "input": input})
        r = self.responses.get(op)
        if callable(r):
            return r(input)
        if r is not None:
            return r
        return {"ok": True, "runtime": input.get("runtime"), "vmId": input.get("vmId")}

    def ops(self) -> List[str]:
        return [c["op"] for c in self.calls if c["op"] not in ("putJson", "getJson")]

    def first(self, op: str) -> Dict[str, Any]:
        return next(c["input"] for c in self.calls if c["op"] == op)


def _load_tool_module(bridge_factory: Callable[[], Any]):
    stub = type(sys)("tool_runtime")
    stub.get_bridge = bridge_factory  # type: ignore[attr-defined]
    sys.modules["tool_runtime"] = stub
    spec = importlib.util.spec_from_file_location(
        "sandbox_pc_tool_" + str(id(bridge_factory)), _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #

def test_invoke_requires_command_for_exec():
    out = _load_tool_module(lambda: _FakeBridge()).invoke("exec", {})
    assert out["ok"] is False and "command" in out["error"]


def test_invoke_requires_file_name_for_write():
    out = _load_tool_module(lambda: _FakeBridge()).invoke("write", {"content": "x"})
    assert out["ok"] is False and "file_name" in out["error"]


def test_no_bridge_reports_error():
    out = _load_tool_module(lambda: None).invoke("start", {"session_id": "s1"})
    assert out["ok"] is False and "bridge" in out["error"]


# --------------------------------------------------------------------------- #
# exec / start / write / suspend / delete -> correct host ops
# --------------------------------------------------------------------------- #

def test_exec_calls_execVm_with_session_vm_id_and_output():
    bridge = _FakeBridge(responses={"execVm": lambda i: {"ok": True, "output": "hello\n",
                                                          "vmId": i["vmId"], "runtime": i["runtime"]}})
    mod = _load_tool_module(lambda: bridge)
    out = mod.invoke("exec", {"command": "echo hello", "runtime": "fire"})

    assert out["ok"] is True
    assert out["output"] == "hello\n"
    assert out["action"] == "exec"
    assert out["vmId"] == "sess-davinci-7"          # derived from bridge.session_id
    assert "execVm" in bridge.ops()
    ev = bridge.first("execVm")
    assert ev["command"] == "echo hello"
    assert ev["machineId"] == "sandbox"             # fixed namespace, not the caller id
    assert ev["vmId"] == "sess-davinci-7"
    assert ev["persistent"] is True
    assert ev["runtime"] == "fire"


def test_exec_runtime_docker_routes_to_docker():
    bridge = _FakeBridge()
    out = _load_tool_module(lambda: bridge).invoke(
        "exec", {"command": "id", "session_id": "abc", "runtime": "docker"})
    ev = bridge.first("execVm")
    assert out["vmId"] == "sess-abc"
    assert ev["runtime"] == "docker"
    assert ev["imageRef"]                           # docker boot needs an image


def test_start_docker_includes_keepalive_command_and_image():
    bridge = _FakeBridge()
    _load_tool_module(lambda: bridge).invoke("start", {"session_id": "s1", "runtime": "docker"})
    rv = bridge.first("runVm")
    assert rv["vmId"] == "sess-s1"
    assert rv["runtime"] == "docker"
    assert "tail -f /dev/null" in rv["command"]     # keep-alive entrypoint
    assert rv["imageRef"]


def test_suspend_calls_terminateVm_purge_false():
    bridge = _FakeBridge()
    out = _load_tool_module(lambda: bridge).invoke("suspend", {"session_id": "s1"})
    tv = bridge.first("terminateVm")
    assert tv["purge"] is False
    assert out["action"] == "suspend"


def test_delete_calls_terminateVm_purge_true():
    bridge = _FakeBridge()
    out = _load_tool_module(lambda: bridge).invoke("delete", {"session_id": "s1"})
    tv = bridge.first("terminateVm")
    assert tv["purge"] is True
    assert out["action"] == "delete"


def test_write_calls_copyToVm():
    bridge = _FakeBridge()
    out = _load_tool_module(lambda: bridge).invoke(
        "write", {"session_id": "s1", "file_name": "a.txt", "content": "hi"})
    cp = bridge.first("copyToVm")
    assert cp["fileName"] == "a.txt" and cp["content"] == "hi"
    assert out["action"] == "write"


# --------------------------------------------------------------------------- #
# SandboxManager unit behaviour
# --------------------------------------------------------------------------- #

def test_manager_session_vm_id_and_resolution():
    from davinci.sandbox_core import SandboxManager, session_vm_id
    assert session_vm_id("davinci-9") == "sess-davinci-9"
    assert session_vm_id("") == "main"
    mgr = SandboxManager(_FakeBridge(session_id=42))
    assert mgr.resolve_session({}) == "davinci-42"
    assert mgr.resolve_session({"session_id": "X"}) == "X"


def test_manager_propagates_host_failure():
    from davinci.sandbox_core import SandboxManager
    bridge = _FakeBridge(responses={"runVm": {"ok": False, "error": "boom"}})
    out = SandboxManager(bridge, default_runtime="fire").start("s1")
    assert out["ok"] is False and out["error"] == "boom"


# --------------------------------------------------------------------------- #
# Runtime session lifecycle (_SessionSandbox + _sandbox_enabled)
# --------------------------------------------------------------------------- #

def test_sandbox_enabled_detects_tool():
    from davinci import caspar_runtime as rt
    assert rt._sandbox_enabled({"caspar__sandbox_pc": {"tool_id": "sandbox_pc"}}) is True
    assert rt._sandbox_enabled({"caspar__web_search": {"tool_id": "web_search"}}) is False


def test_session_sandbox_starts_and_suspends_via_host_ops():
    from davinci import caspar_runtime as rt
    bridge = _FakeBridge()
    with rt._SessionSandbox(bridge, True, "sid-1"):
        assert bridge.ops() == ["runVm"]            # started on entry
        assert bridge.first("runVm")["vmId"] == "sess-sid-1"
    assert bridge.ops() == ["runVm", "terminateVm"]  # suspended on exit
    tv = bridge.first("terminateVm")
    assert tv["purge"] is False


def test_session_sandbox_noop_when_disabled():
    from davinci import caspar_runtime as rt
    bridge = _FakeBridge()
    with rt._SessionSandbox(bridge, False, "x"):
        pass
    assert bridge.calls == []


def test_session_sandbox_noop_without_bridge():
    from davinci import caspar_runtime as rt
    with rt._SessionSandbox(None, True, "x"):
        pass  # must not raise
