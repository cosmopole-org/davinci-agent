"""sandbox_pc tool — per-session Firecracker sandbox proxy.

The actual sandbox is a WASM creature (``decillionai-server/creatures/sandbox``)
running on Caspar that drives the ``fire`` VM runtime through the Caspar host
APIs (``runVm``/``execVm``/``terminateVm``). This tool is a thin client that
signals that creature over the docker-host bridge and returns its reply (which
includes the shell output from inside the fire VM).

Lifecycle, end to end:

* davinci registers ``caspar__sandbox_pc`` like any other MCP tool. When the
  agent calls ``exec`` mid-loop, ``invoke`` here is run.
* ``invoke`` signals the sandbox WASM creature on ``creatures/signal`` with an
  envelope carrying ``{action, payload, correlationId, reply_to}``.
* The sandbox creature calls ``execVm`` on the Caspar host. The host wakes a
  suspended VM transparently (wake-on-exec) on its persistent disk under
  ``{STORAGE_ROOT_PATH}/vms/<machine>/<vm>``, runs the command, and returns the
  shell output.
* The sandbox creature signals ``creatures/signal/result`` back to davinci,
  matched by ``correlationId``. We return it to the agent.

While the session is actively serviced, davinci pings ``start`` to keep the VM
warm; when the agent's reply is delivered and the session goes idle, davinci
issues ``suspend`` so the VM is turned off (almost terminated) but its disk —
the installed software and data — survives until the next prompt wakes it.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# Bridge handle. The tool runtime opens the gateway connection before invoke()
# is called and exposes it via tool_runtime.get_bridge(); we re-import lazily so
# this module also works when imported standalone in tests.
# --------------------------------------------------------------------------- #

def _get_bridge():
    try:
        import tool_runtime  # type: ignore
        return tool_runtime.get_bridge()
    except Exception:
        return None


# Target wasm sandbox creature. Provided by the deploy harness via env so a
# single tool image works against any Caspar deployment; the legacy default
# matches the creature name used in decillionai-server's build/deploy pipeline.
_DEFAULT_SANDBOX_TARGET = os.environ.get("SANDBOX_PC_TARGET", "sandbox")


# --------------------------------------------------------------------------- #
# Per-instance signaling: send -> wait for correlation-id reply
# --------------------------------------------------------------------------- #

_waiters: Dict[str, threading.Event] = {}
_results: Dict[str, Any] = {}
_waiters_lock = threading.Lock()
_handler_installed = False


def _install_reply_handler(bridge) -> None:
    """Register an on_signal handler that completes waiters by correlationId.

    Idempotent: only the first call wires the handler.
    """
    global _handler_installed
    if _handler_installed or bridge is None:
        return

    def _handle(frame: Any) -> None:
        # Frames arrive as already-parsed dicts; the sandbox creature replies
        # with ``signalUser("creatures/signal/result", ..., packet=<json>)`` so
        # the payload sits under either ``packet`` (string) or ``data``.
        try:
            data = frame
            if isinstance(frame, dict):
                if isinstance(frame.get("packet"), str):
                    try:
                        data = json.loads(frame["packet"])
                    except Exception:
                        data = frame
                elif "data" in frame and isinstance(frame["data"], str):
                    try:
                        data = json.loads(frame["data"])
                    except Exception:
                        data = frame
            if not isinstance(data, dict):
                return
            cid = data.get("correlationId")
            if not cid:
                return
            with _waiters_lock:
                ev = _waiters.get(cid)
                if ev is None:
                    return
                _results[cid] = data
                ev.set()
        except Exception:
            return

    try:
        bridge.on_signal(_handle)
        _handler_installed = True
    except Exception:
        # Even without the handler we can still send fire-and-wait via polling
        # on the bridge's own correlation map; the wait below degrades cleanly.
        pass


def _session_id(payload: Dict[str, Any]) -> str:
    """Resolve the davinci session id. Per-session VMs key off this value, so we
    fall back through the conventional sources before generating a stable id
    from the bridge handshake (so a single agent run always hits the same VM).
    """
    for k in ("session_id", "sessionId", "session", "vm_id", "vmId"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    bridge = _get_bridge()
    if bridge is not None:
        sid = getattr(bridge, "session_id", None)
        if sid:
            return f"davinci-{sid}"
    return "davinci-default"


def _signal_sandbox(action: str, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
    """Send one action to the sandbox creature and await its result."""
    bridge = _get_bridge()
    if bridge is None:
        return {"ok": False, "error": "bridge unavailable; cannot reach sandbox creature"}
    _install_reply_handler(bridge)

    target = payload.get("sandbox_target") or _DEFAULT_SANDBOX_TARGET
    reply_to = getattr(bridge, "machine_id", "") or getattr(bridge, "creature_id", "") or "davinci"
    correlation_id = uuid.uuid4().hex
    session_id = _session_id(payload)

    # Sandbox VM runtime: "fire" (Firecracker microVM) or "docker" (gVisor
    # container). The sandbox creature drives both; the default mirrors the
    # creature's own default ("fire"). A caller can pin "docker" when KVM is
    # unavailable.
    inner_payload = {k: v for k, v in payload.items() if k not in ("sandbox_target",)}
    inner_payload["sessionId"] = session_id
    runtime = payload.get("runtime") or os.environ.get("SANDBOX_PC_RUNTIME")
    if runtime:
        inner_payload["runtime"] = runtime

    # Match the envelope shape the sandbox creature's unwrapSignal() expects.
    envelope: Dict[str, Any] = {
        "kind": "invoke",
        "correlationId": correlation_id,
        "reply_to": reply_to,
        "tool_id": "sandbox_pc",
        "function": action,
        "action": action,
        "payload": inner_payload,
    }

    ev = threading.Event()
    with _waiters_lock:
        _waiters[correlation_id] = ev
    try:
        bridge.signal_user("creatures/signal", str(target), envelope)
    except Exception as exc:
        with _waiters_lock:
            _waiters.pop(correlation_id, None)
        return {"ok": False, "error": f"signal failed: {exc!r}"}

    delivered = ev.wait(timeout)
    with _waiters_lock:
        _waiters.pop(correlation_id, None)
        result = _results.pop(correlation_id, None)
    if not delivered:
        return {"ok": False, "error": f"sandbox reply timed out after {timeout:.0f}s",
                "session_id": session_id}
    if not isinstance(result, dict):
        return {"ok": False, "error": "malformed sandbox reply", "raw": result,
                "session_id": session_id}
    result.setdefault("session_id", session_id)
    return result


# --------------------------------------------------------------------------- #
# Public surface (tool registry calls invoke(function_name, payload))
# --------------------------------------------------------------------------- #

_ACTION_DEFAULT_TIMEOUTS: Dict[str, float] = {
    "exec": 75.0,
    "start": 30.0,
    "wake": 30.0,
    "resume": 30.0,
    "write": 30.0,
    "suspend": 15.0,
    "stop": 15.0,
    "delete": 30.0,
    "destroy": 30.0,
    "status": 10.0,
}


def _normalize_action(function_name: str, payload: Dict[str, Any]) -> str:
    explicit = payload.get("action")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if function_name and function_name not in ("invoke", ""):
        return function_name
    if payload.get("command"):
        return "exec"
    return "status"


def invoke(function_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(payload or {})
    action = _normalize_action(function_name, payload)

    if action == "exec" and not payload.get("command"):
        return {"ok": False, "error": "no 'command' provided", "function": function_name}
    if action == "write" and not payload.get("content"):
        return {"ok": False, "error": "no 'content' provided", "function": function_name}

    timeout = payload.pop("timeout_seconds", None)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        timeout = _ACTION_DEFAULT_TIMEOUTS.get(action, 60.0)
    timeout = float(max(5.0, min(float(timeout), 600.0)))

    result = _signal_sandbox(action, payload, timeout=timeout)
    result.setdefault("function", function_name)
    result.setdefault("action", action)
    return result
