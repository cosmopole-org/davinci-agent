"""Per-session sandbox VM management over the Caspar vmm host APIs.

A *sandbox VM* is one Caspar VM bonded to a single davinci session. It can be a
``fire`` Firecracker microVM or a ``docker`` gVisor container — both expose the
same lifecycle (start / exec / write / suspend / delete) on the node, with a
persistent, non-escapable disk under ``{storage}/vms/<vm>``.

This module manages those VMs **directly** through the node's vmm host
functions over the docker-host bridge — ``runVm`` / ``execVm`` /
``terminateVm`` / ``copyToVm``. There is no intermediary creature: the davinci
``sandbox_pc`` tool (which itself runs as a Caspar ``docker`` creature) and the
davinci runtime's session-lifecycle hook both use this one class, so a session's
VM is created, woken, used, and suspended through a single code path.

Addressing. A sandbox VM is keyed by ``(machineId, vmId)``:

* ``machineId`` is a fixed namespace constant — deliberately NOT the calling
  creature's identity — so the tool and the lifecycle hook (which run in
  different creatures, hence have different bridge identities) resolve the
  *same* VM.
* ``vmId`` encodes the session (``sess-<session>``). Each session therefore
  gets exactly one VM, and a later prompt in that session resumes it. The
  node keys the persistent disk on ``vmId`` alone, so the bonding survives
  suspend/resume.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

# Fixed namespace for sandbox-VM container/process keys. It must be identical
# everywhere a session's VM is addressed, so it is a constant rather than the
# caller's creature id (the tool and the lifecycle hook are different
# creatures). Overridable per-deployment via env, but the default is shared.
SANDBOX_MACHINE_ID = os.environ.get("SANDBOX_PC_MACHINE", "sandbox")

# How long the node keeps a sandbox VM alive before its safety timeout fires.
# The session lifecycle suspends the VM well before this; it is only a backstop.
DEFAULT_KEEPALIVE_SECONDS = 1800

# Default base image for the docker runtime when first booting a sandbox.
DEFAULT_DOCKER_IMAGE = os.environ.get("SANDBOX_PC_IMAGE", "alpine:latest")

# Keep-alive entrypoint so a docker sandbox container stays up (idle) until we
# explicitly suspend or delete it, letting later `exec`s attach to it.
_DOCKER_KEEPALIVE_CMD = "sh -lc 'mkdir -p /data && tail -f /dev/null'"


def sanitize_id(raw: str) -> str:
    """Reduce an id to a safe path/key component (mirrors the host guard)."""
    out = re.sub(r"[^A-Za-z0-9_-]", "_", raw or "")
    return out or "default"


def session_vm_id(session_id: str) -> str:
    """Map a davinci session id to its one-and-only sandbox VM id."""
    sid = sanitize_id(session_id) if session_id else ""
    return ("sess-" + sid) if sid else "main"


class SandboxManager:
    """Drives a session's sandbox VM through the vmm host functions."""

    def __init__(
        self,
        bridge: Any,
        *,
        default_runtime: Optional[str] = None,
        machine_id: str = SANDBOX_MACHINE_ID,
    ) -> None:
        self.bridge = bridge
        self.machine_id = machine_id
        # fire when KVM is available; docker otherwise. A per-deployment default
        # can be pinned via SANDBOX_PC_RUNTIME so the tool and the lifecycle
        # hook agree without the caller having to pass it every time.
        self.default_runtime = (
            default_runtime or os.environ.get("SANDBOX_PC_RUNTIME") or "fire"
        )

    # -- internals ---------------------------------------------------------
    def _runtime(self, runtime: Optional[str]) -> str:
        rt = (runtime or self.default_runtime).strip().lower()
        return rt if rt in ("fire", "docker") else "fire"

    def _base(self, session_id: str, runtime: Optional[str], keep_alive: int) -> Dict[str, Any]:
        """The host-op fields shared by every lifecycle call for a VM."""
        rt = self._runtime(runtime)
        return {
            "runtime": rt,
            "machineId": self.machine_id,
            "targetCreatureId": self.machine_id,
            "vmId": session_vm_id(session_id),
            "persistent": True,
            "entityId": "sandbox",
            "containerName": "session",
            "resources": {"maxExecTimeSeconds": keep_alive},
        }

    def _call(self, op: str, inp: Dict[str, Any]) -> Dict[str, Any]:
        if self.bridge is None:
            return {"ok": False, "error": "bridge unavailable; cannot reach vmm host"}
        try:
            res = self.bridge.call(op, inp)
        except Exception as exc:  # noqa: BLE001 — surface, never raise into the agent
            return {"ok": False, "error": f"host call {op!r} failed: {exc!r}"}
        if isinstance(res, dict):
            return res
        if isinstance(res, str):
            try:
                return json.loads(res)
            except Exception:  # noqa: BLE001
                return {"ok": True, "raw": res}
        return {"ok": bool(res), "raw": res}

    def _state_key(self, vm_id: str) -> str:
        return f"Json::Sandbox::{self.machine_id}::{vm_id}"

    def _record(self, vm_id: str, runtime: str, last_action: str) -> None:
        """Persist a tiny state record (best-effort) for `status`/auditing."""
        self._call("putJson", {
            "key": self._state_key(vm_id),
            "path": "state",
            "data": {"vmId": vm_id, "machineId": self.machine_id,
                     "runtime": runtime, "lastAction": last_action},
            "merge": True,
        })

    def resolve_session(self, payload: Dict[str, Any]) -> str:
        """Resolve the session id from a payload, falling back to the bridge
        handshake session so a single agent run always hits the same VM."""
        for k in ("session_id", "sessionId", "session", "vm_id", "vmId"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        sid = getattr(self.bridge, "session_id", None)
        return f"davinci-{sid}" if sid else "davinci-default"

    # -- lifecycle ---------------------------------------------------------
    def start(self, session_id: str, *, runtime: Optional[str] = None,
              image: Optional[str] = None,
              keep_alive: int = DEFAULT_KEEPALIVE_SECONDS) -> Dict[str, Any]:
        """Boot or resume the session's VM on its persistent disk."""
        inp = self._base(session_id, runtime, keep_alive)
        rt = inp["runtime"]
        if rt == "docker":
            inp["imageRef"] = image or DEFAULT_DOCKER_IMAGE
            inp["command"] = _DOCKER_KEEPALIVE_CMD
        elif image:
            inp["image"] = image
        host = self._call("runVm", inp)
        self._record(inp["vmId"], rt, "start")
        return self._wrap("start", inp, host, status=host.get("status", "running"))

    def exec(self, session_id: str, command: str, *, runtime: Optional[str] = None,
             keep_alive: int = DEFAULT_KEEPALIVE_SECONDS) -> Dict[str, Any]:
        """Run a shell command in the session VM (host wakes it if suspended)."""
        if not command:
            return {"ok": False, "error": "command is required", "action": "exec"}
        inp = self._base(session_id, runtime, keep_alive)
        inp["command"] = command
        if inp["runtime"] == "docker":
            inp["imageRef"] = DEFAULT_DOCKER_IMAGE  # used only if a cold boot is needed
        host = self._call("execVm", inp)
        self._record(inp["vmId"], inp["runtime"], "exec")
        out = self._wrap("exec", inp, host, status="running")
        out["output"] = host.get("output", "")
        return out

    def write(self, session_id: str, file_name: str, content: str, *,
              target_path: str = "/data", runtime: Optional[str] = None) -> Dict[str, Any]:
        """Drop a file into the session VM."""
        if not file_name:
            return {"ok": False, "error": "file_name is required", "action": "write"}
        inp = self._base(session_id, runtime, DEFAULT_KEEPALIVE_SECONDS)
        inp.update({"fileName": file_name, "targetPath": target_path, "content": content or ""})
        if inp["runtime"] == "docker":
            inp["imageRef"] = DEFAULT_DOCKER_IMAGE
        host = self._call("copyToVm", inp)
        return self._wrap("write", inp, host)

    def suspend(self, session_id: str, *, runtime: Optional[str] = None) -> Dict[str, Any]:
        """Turn the VM off but KEEP its disk — the idle state between prompts."""
        inp = self._base(session_id, runtime, DEFAULT_KEEPALIVE_SECONDS)
        inp["purge"] = False
        host = self._call("terminateVm", inp)
        self._record(inp["vmId"], inp["runtime"], "suspend")
        return self._wrap("suspend", inp, host, status="suspended")

    def delete(self, session_id: str, *, runtime: Optional[str] = None) -> Dict[str, Any]:
        """Turn the VM off AND purge its disk — destroys the sandbox."""
        inp = self._base(session_id, runtime, DEFAULT_KEEPALIVE_SECONDS)
        inp["purge"] = True
        host = self._call("terminateVm", inp)
        self._record(inp["vmId"], inp["runtime"], "delete")
        return self._wrap("delete", inp, host, status="deleted")

    def status(self, session_id: str, *, runtime: Optional[str] = None) -> Dict[str, Any]:
        """Report the session VM's last-known state from host storage."""
        vm_id = session_vm_id(session_id)
        rec = self._call("getJson", {"key": self._state_key(vm_id), "path": "state"})
        return {"ok": True, "action": "status", "vmId": vm_id,
                "runtime": self._runtime(runtime), "state": rec}

    # -- dispatch (used by the tool's invoke) ------------------------------
    def dispatch(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = self.resolve_session(payload)
        runtime = payload.get("runtime")
        if action in ("exec", "execCommand", "run"):
            return self.exec(session_id, payload.get("command", ""), runtime=runtime)
        if action in ("start", "resume", "wake", "runPc"):
            return self.start(session_id, runtime=runtime, image=payload.get("image"))
        if action in ("write", "copyTo"):
            return self.write(
                session_id,
                payload.get("file_name") or payload.get("fileName") or "",
                payload.get("content") or "",
                target_path=payload.get("target_path") or payload.get("targetPath") or "/data",
                runtime=runtime,
            )
        if action in ("suspend", "stop", "sleep"):
            return self.suspend(session_id, runtime=runtime)
        if action in ("delete", "destroy", "terminate", "stopPc"):
            return self.delete(session_id, runtime=runtime)
        if action == "status":
            return self.status(session_id, runtime=runtime)
        return {"ok": False, "error": f"unknown sandbox action: {action}", "action": action}

    @staticmethod
    def _wrap(action: str, inp: Dict[str, Any], host: Dict[str, Any],
              *, status: Optional[str] = None) -> Dict[str, Any]:
        out = {
            "ok": bool(host.get("ok", False)),
            "action": action,
            "vmId": inp["vmId"],
            "runtime": inp["runtime"],
            "host": host,
        }
        if not out["ok"] and host.get("error"):
            out["error"] = host["error"]
        if status is not None and out["ok"]:
            out["status"] = host.get("status", status)
        return out
