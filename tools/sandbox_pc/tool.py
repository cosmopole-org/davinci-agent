"""sandbox_pc tool — per-session Firecracker/docker sandbox.

This tool gives the agent a private, persistent Linux sandbox per session — one
Caspar VM (``fire`` Firecracker microVM or ``docker`` gVisor container) bonded
to the session, whose installed software and data survive between turns.

The tool itself runs as a Caspar ``docker`` creature. It manages the sandbox VM
**directly** through the node's vmm host functions over the docker-host bridge
(``runVm`` / ``execVm`` / ``terminateVm`` / ``copyToVm``) — there is no
intermediary creature to proxy through. All of that logic lives in the shared
:class:`davinci.sandbox_core.SandboxManager`, which the davinci runtime's
session-lifecycle hook uses too, so the VM is addressed identically from both.

Lifecycle:

* The agent calls ``exec`` to run a command; the host transparently wakes a
  suspended VM (wake-on-exec) on its persistent disk under ``{storage}/vms/<vm>``.
* While the session is actively serviced the runtime keeps the VM warm; when the
  reply is delivered the runtime ``suspend``s it — off, but its disk (installed
  software + data) survives until the next prompt wakes it.

Functions: ``exec`` · ``start`` · ``write`` · ``suspend`` · ``delete`` · ``status``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# The shared manager is shipped into the tool image as a top-level module by the
# deploy harness; fall back to the package path for in-repo execution / tests.
try:  # pragma: no cover - import shim
    from sandbox_core import SandboxManager  # type: ignore
except Exception:  # pragma: no cover
    from davinci.sandbox_core import SandboxManager  # type: ignore


def _get_bridge():
    """The docker-host bridge the tool runtime opened before invoke()."""
    try:
        import tool_runtime  # type: ignore
        return tool_runtime.get_bridge()
    except Exception:
        return None


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
    manager = SandboxManager(_get_bridge())
    result = manager.dispatch(action, payload)
    result.setdefault("function", function_name)
    result.setdefault("action", action)
    return result
