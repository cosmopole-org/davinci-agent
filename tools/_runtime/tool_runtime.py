"""Generic Davinci tool-creature runtime.

Every Davinci tool is deployed as its own Caspar ``docker`` creature. When the
node signals it (via ``/programs/runEntity``), the node uploads the signal to
``/app/input/task.json`` and captures this process's stdout as VM logs.

Contract:
    input  : /app/input/task.json = {"tool_id","function","payload"}
    output : a single line  ``TOOL_RESPONSE <json>``  on stdout

The runtime dispatches to a built-in handler for well-known tools (so the demo
does real work where it safely can), and otherwise falls back to the tool's own
``invoke`` function deployed alongside it.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

INPUT_DIR = os.environ.get("DAVINCI_INPUT_DIR", "/app/input")
TOOL_ID = os.environ.get("TOOL_ID", "")


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
# Built-in handlers (lightweight but real where safe)
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

    # Bound the whole call (DNS resolution can hang past the socket timeout in a
    # sandbox with no resolver); never let a tool creature block its response.
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


def _dispatch(tool_id: str, function: str, payload: dict) -> dict:
    handler = _BUILTINS.get(tool_id)
    if handler:
        try:
            return handler(payload)  # type: ignore[arg-type]
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    # Fall back to the deployed tool.py invoke(), supporting both signatures.
    try:
        import tool  # type: ignore
        if hasattr(tool, "invoke"):
            try:
                return tool.invoke(function, payload)  # type: ignore[misc]
            except TypeError:
                return tool.invoke(payload)  # type: ignore[misc]
    except Exception:
        pass
    return _h_echo(tool_id, payload)


def main() -> int:
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
    print("TOOL_RESPONSE " + json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
