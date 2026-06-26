"""Generic Davinci tool-creature runtime.

Every Davinci tool is deployed as its own Caspar ``docker`` creature. When the
node signals it (via ``/programs/runEntity``), the node uploads the signal to
``/app/input/task.json`` and captures this process's stdout as VM logs.

Contract:
    input  : /app/input/task.json = {"tool_id","function","payload"}
    output : a single line  ``TOOL_RESPONSE <json>``  on stdout

Dispatch order:
    1. The tool's own deployed ``tool.py`` ``invoke`` (the real implementation,
       shipped alongside this runtime with its own dependencies installed in the
       tool image). This is what runs in production.
    2. A small built-in fallback for a few well-known tools, so the runtime is
       still useful when no ``tool.py`` is present (e.g. unit tests).
    3. An echo, so an unknown tool never hard-fails.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

INPUT_DIR = os.environ.get("DAVINCI_INPUT_DIR", "/app/input")
TOOL_ID = os.environ.get("TOOL_ID", "")


# --------------------------------------------------------------------------- #
# Docker-host bridge gateway
#
# When this tool runs as a gateway-managed docker creature the node injects
# CASPAR_GATEWAY_* env vars and the bridge client is shipped alongside this
# runtime (as ``caspar_bridge.py``). The bridge is the tool's only route to the
# outside world: tool implementations reach the node's HTTP/DB host functions
# through ``get_bridge()`` and the result is signalled back over the same
# connection. Everything here is env-guarded so the runtime stays usable (and
# unit-testable) with no gateway present.
# --------------------------------------------------------------------------- #
try:  # shipped into the tool image as a top-level module by the deploy harness
    import caspar_bridge as _bridge_mod  # type: ignore
except Exception:  # pragma: no cover - fallback for in-repo execution / tests
    try:
        from davinci import caspar_bridge as _bridge_mod  # type: ignore
    except Exception:
        _bridge_mod = None  # type: ignore

_BRIDGE = None  # type: ignore


def get_bridge():
    """Return the connected bridge client, or ``None`` when not gateway-managed."""
    return _BRIDGE


def _connect_bridge():
    global _BRIDGE
    if _BRIDGE is not None or _bridge_mod is None:
        return _BRIDGE
    # When no gateway is configured (CASPAR_GATEWAY_HOST unset — local/unit
    # tests) ``bridge_from_env`` returns None *without* connecting, so we fall
    # straight through to offline mode below. When a gateway IS configured we
    # retry a few times: the node binds the container's source IP to its identity
    # (``register_vm_container``) right after starting the container, so a
    # creature that connects in the first moments can momentarily lose the HELLO
    # identity race ("could not identify a docker creature for source ip ...").
    # A handful of short retries closes that window so a serving tool reliably
    # reaches TOOL_SERVE_READY instead of silently dropping to one-shot mode.
    attempts = int(os.environ.get("TOOL_BRIDGE_CONNECT_ATTEMPTS", "5"))
    last_exc = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            _BRIDGE = _bridge_mod.bridge_from_env()
            return _BRIDGE
        except Exception as exc:  # noqa: BLE001 — never block the tool on bridge setup
            last_exc = exc
            if attempt < attempts:
                time.sleep(min(2.0, 0.5 * attempt))
    print(f"TOOL_BRIDGE {json.dumps({'connect_error': repr(last_exc)[:160], 'attempts': attempts})}",
          flush=True)
    _BRIDGE = None
    return _BRIDGE


def _reply_over_bridge(signal: dict, response: dict) -> None:
    """If the request carried reply metadata, push the result back over the
    gateway so the caller is notified through the connection (not just logs)."""
    bridge = _BRIDGE
    if bridge is None:
        return
    payload = signal.get("payload") or {}
    reply_to = signal.get("reply_to") or payload.get("reply_to") or signal.get("userId")
    correlation_id = signal.get("correlationId") or payload.get("correlationId")
    if not reply_to:
        return
    # Reply on the "creatures/signal" key so it passes the caller's machine
    # listener and is pushed onto the caller's gateway connection; the structured
    # packet lets the caller match it to the originating request.
    try:
        bridge.signal_user(
            "creatures/signal",
            str(reply_to),
            {"kind": "tools/result", "correlationId": correlation_id, "result": response},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"TOOL_BRIDGE {json.dumps({'reply_error': repr(exc)[:160]})}", flush=True)


def _read_signal() -> dict:
    path = os.path.join(INPUT_DIR, "task.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"tool_id": TOOL_ID, "function": "invoke", "payload": {}}


# --------------------------------------------------------------------------- #
# Built-in fallback handlers (used only when no real tool.py is deployed)
# --------------------------------------------------------------------------- #

def _h_python_exec(payload: dict) -> dict:
    code = str(payload.get("code") or "result = 6 * 7")
    scope: dict = {}
    try:
        exec(code, {"__builtins__": __builtins__}, scope)  # noqa: S102 - sandboxed by gVisor
        return {"ok": True, "result": scope.get("result"), "vars": list(scope.keys())}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _h_http(payload: dict) -> dict:
    url = payload.get("url") or "https://example.com"
    out: dict = {}

    def _fetch():
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                body = resp.read(2048).decode("utf-8", "replace")
            out.update({"ok": True, "url": url, "status": resp.status,
                        "bytes": len(body), "preview": body[:200]})
        except Exception as exc:
            out.update({"ok": False, "url": url, "error": str(exc)})

    import threading
    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(7)
    if not out:
        return {"ok": False, "url": url, "error": "network unavailable (sandbox); request timed out",
                "query": payload.get("query") or payload.get("task")}
    return out


def _h_vector_search(payload: dict) -> dict:
    query = str(payload.get("query") or payload.get("task") or "")
    corpus = payload.get("corpus") or [
        "Davinci is an orchestration-first enterprise agent.",
        "Caspar runs creatures across six VM runtimes.",
        "Tool creatures respond to signals over the action protocol.",
    ]
    q = set(query.lower().split())
    ranked = sorted(corpus, key=lambda d: -len(q & set(d.lower().split())))
    return {"ok": True, "query": query, "top": ranked[:2]}


def _h_echo(tool_id: str, payload: dict) -> dict:
    return {"ok": True, "tool_id": tool_id, "echo": payload}


_BUILTINS = {
    "python_exec": _h_python_exec,
    "web_search": _h_http,
    "fetch_url": _h_http,
    "vector_search": _h_vector_search,
}


def _load_tool_module():
    """Import the deployed ``tool.py`` (the real per-tool implementation)."""
    try:
        import tool  # type: ignore
        return tool
    except Exception:
        return None


def _call_invoke(impl, function: str, payload: dict) -> dict:
    """Call a tool's invoke() supporting both (function, payload) and (payload)."""
    try:
        return impl.invoke(function, payload)  # type: ignore[misc]
    except TypeError:
        return impl.invoke(payload)  # type: ignore[misc]


def _dispatch(tool_id: str, function: str, payload: dict) -> dict:
    # 1. Prefer the tool's real implementation when it ships with the image.
    impl = _load_tool_module()
    if impl is not None and hasattr(impl, "invoke"):
        try:
            return _call_invoke(impl, function, payload)
        except Exception:
            return {"ok": False, "error": traceback.format_exc().splitlines()[-1]}

    # 2. Built-in fallback for well-known tools.
    handler = _BUILTINS.get(tool_id)
    if handler:
        try:
            return handler(payload)  # type: ignore[arg-type]
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # 3. Echo so unknown tools never hard-fail.
    return _h_echo(tool_id, payload)


def _extract_invoke(data: dict) -> dict:
    """Normalise the SIGNAL payload into an ``invoke`` packet.

    Two delivery shapes reach a serving tool, both over the signaling API:

    * **creature → creature** — a sibling creature (e.g. Davinci) calls
      ``signalUser`` with the invoke packet as the signal value, so ``data`` *is*
      the packet (``data["kind"] == "invoke"``).
    * **external client → creature** — a ``/creatures/signal`` (pvp) is wrapped by
      the node as ``StoresSend{action, user, data:"<json>", entityId}``; the real
      packet is the JSON string in ``data["data"]``.
    """
    if not isinstance(data, dict):
        return {}
    if data.get("kind") == "invoke" or "tool_id" in data or "function" in data:
        return data
    inner = data.get("data")
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except Exception:  # noqa: BLE001
            return {}
    return inner if isinstance(inner, dict) else {}


def _handle_invoke(bridge, packet: dict) -> None:
    """Run one tool invocation and signal the result back to the caller.

    Runs on a worker thread — never on the bridge reader thread — because the
    reply is a host call whose response is read by that same reader thread; doing
    it inline would deadlock.
    """
    tool_id = packet.get("tool_id") or TOOL_ID or "unknown"
    function = packet.get("function", "invoke")
    payload = packet.get("payload") or {}
    print(f"TOOL_BOOT {json.dumps({'tool_id': tool_id, 'function': function, 'ts': time.time()})}", flush=True)
    try:
        result = _dispatch(tool_id, function, payload)
    except Exception:
        result = {"ok": False, "error": traceback.format_exc().splitlines()[-1]}
    response = {"tool_id": tool_id, "function": function, "result": result}
    print("TOOL_RESPONSE " + json.dumps(response, default=str), flush=True)
    reply_to = packet.get("reply_to") or payload.get("reply_to")
    correlation_id = packet.get("correlationId") or payload.get("correlationId")
    if bridge is not None and reply_to:
        try:
            bridge.signal_user(
                "creatures/signal", str(reply_to),
                {"kind": "tools/result", "correlationId": correlation_id, "result": response})
        except Exception as exc:  # noqa: BLE001
            print(f"TOOL_BRIDGE {json.dumps({'reply_error': repr(exc)[:160]})}", flush=True)


def _serve(bridge) -> int:
    """Run the tool as a long-lived standalone creature.

    The tool VM is started once (via ``runEntity``) and then stays alive,
    receiving work purely as pushed signals over the docker-host gateway and
    replying over the same channel. It never reads a task file and is never
    cold-spawned per call — Davinci and other creatures reach it through the
    Caspar signaling API.
    """
    import threading

    tool_id = TOOL_ID or "unknown"
    state = {"last": time.time(), "served": 0}
    idle_timeout = float(os.environ.get("TOOL_SERVE_IDLE", "600"))

    def on_signal(key: str, data) -> None:
        if key != "creatures/signal":
            return
        packet = _extract_invoke(data if isinstance(data, dict) else {})
        # Ignore our own result echoes and anything that isn't an invocation.
        if not packet or packet.get("kind") == "tools/result":
            return
        if not (packet.get("tool_id") or packet.get("function") or packet.get("payload")):
            return
        state["last"] = time.time()
        state["served"] += 1
        threading.Thread(target=_handle_invoke, args=(bridge, packet), daemon=True).start()

    bridge.on_signal(on_signal)
    print("TOOL_SERVE_READY " + json.dumps(
        {"tool_id": tool_id, "machine_id": getattr(bridge, "machine_id", ""),
         "program_id": getattr(bridge, "program_id", ""), "ts": time.time()}), flush=True)

    # Stay alive serving signals until the node terminates the VM (its
    # max_exec_seconds) or no work has arrived for the idle window.
    while time.time() - state["last"] < idle_timeout:
        time.sleep(2)
    print("TOOL_SERVE_EXIT " + json.dumps({"tool_id": tool_id, "served": state["served"]}), flush=True)
    try:
        bridge.close()
    except Exception:  # noqa: BLE001
        pass
    return 0


def _run_once_offline() -> int:
    """One-shot fallback for offline/local execution with no gateway (unit tests):
    read a task file if present, dispatch once, print the response."""
    signal = _read_signal()
    tool_id = signal.get("tool_id") or TOOL_ID or "unknown"
    function = signal.get("function", "invoke")
    payload = signal.get("payload") or {}
    print(f"TOOL_BOOT {json.dumps({'tool_id': tool_id, 'function': function, 'ts': time.time()})}", flush=True)
    try:
        result = _dispatch(tool_id, function, payload)
    except Exception:
        result = {"ok": False, "error": traceback.format_exc().splitlines()[-1]}
    response = {"tool_id": tool_id, "function": function, "result": result}
    print("TOOL_RESPONSE " + json.dumps(response, default=str), flush=True)
    return 0


def main() -> int:
    # The single gateway connection is the tool's only channel to the outside
    # world. When present, the tool runs as a long-lived standalone creature that
    # is driven purely through the signaling API. With no gateway (local/unit
    # tests) it falls back to a single offline dispatch.
    bridge = _connect_bridge()
    if bridge is not None:
        print(f"TOOL_BRIDGE {json.dumps({'connected': True, 'session': bridge.session_id})}", flush=True)
        return _serve(bridge)
    return _run_once_offline()


if __name__ == "__main__":
    sys.exit(main())
