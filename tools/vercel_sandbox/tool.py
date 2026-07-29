"""vercel_sandbox tool creature — one Vercel Sandbox microVM per Decillion space.

Deployed as its own Caspar ``docker`` creature (like every other Davinci tool)
and driven purely through the Caspar signalling API: Nest signals it when a
space is created/deleted, and the space's agents signal it — through Davinci's
bridge executor — whenever they want to run something.

State lives entirely on Vercel's side. The binding between a Decillion space and
its sandbox is the sandbox's **name**, derived deterministically from the space
id (:func:`sandbox_name`), so any creature that knows the space id can address
the same sandbox without a local database. Named sandboxes snapshot themselves
on shutdown, so a space's filesystem survives the VM being stopped and resumed.

Vercel REST API used (v2 named sandboxes + sessions)::

    POST   /v2/sandboxes                                  create (named)
    GET    /v2/sandboxes/{name}?resume=true               get / resume -> session
    DELETE /v2/sandboxes/{name}                           destroy
    POST   /v2/sandboxes/sessions/{sid}/cmd               exec (wait+logs ND-JSON)
    POST   /v2/sandboxes/sessions/{sid}/fs/write          upload a .tar.gz
    POST   /v2/sandboxes/sessions/{sid}/fs/read           download a file
    POST   /v2/sandboxes/sessions/{sid}/fs/mkdir          create a directory
    POST   /v2/sandboxes/sessions/{sid}/stop              stop the session

Credentials come from the container environment only — never from the signal
payload, so a prompt-injected agent cannot swap the token or the target team:

    VERCEL_TOKEN        (or VERCEL_API_TOKEN / VERCEL_ACCESS_TOKEN)   required
    VERCEL_TEAM_ID      team the sandboxes are billed to              optional
    VERCEL_PROJECT_ID   project that owns the named sandboxes         optional
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import tarfile
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

API_BASE = os.environ.get("VERCEL_API_BASE", "https://api.vercel.com").rstrip("/")
HTTP_TIMEOUT = float(os.environ.get("VERCEL_SANDBOX_HTTP_TIMEOUT", "60"))
# Wall-clock cap for one `exec`, independent of the HTTP read timeout.
EXEC_TIMEOUT_MS = int(os.environ.get("VERCEL_SANDBOX_EXEC_TIMEOUT_MS", "300000"))
# Command output is fed back into an LLM context — cap it hard.
MAX_OUTPUT_CHARS = int(os.environ.get("VERCEL_SANDBOX_MAX_OUTPUT", "60000"))
MAX_READ_BYTES = int(os.environ.get("VERCEL_SANDBOX_MAX_READ_BYTES", "1000000"))

NAME_PREFIX = os.environ.get("VERCEL_SANDBOX_PREFIX", "decillion")
DEFAULT_RUNTIME = os.environ.get("VERCEL_SANDBOX_RUNTIME", "node24")
DEFAULT_TIMEOUT_MS = int(os.environ.get("VERCEL_SANDBOX_TIMEOUT_MS", str(45 * 60 * 1000)))
DEFAULT_VCPUS = int(os.environ.get("VERCEL_SANDBOX_VCPUS", "2"))

_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]+")

# session id cache: name -> (session_id, monotonic deadline). A stopped session
# is only detected when a call against it 404/410s, so the TTL is short and every
# caller re-resolves through `_session` on failure.
_SESSIONS: Dict[str, Tuple[str, float]] = {}
_SESSION_TTL = float(os.environ.get("VERCEL_SANDBOX_SESSION_TTL", "120"))


class SandboxError(RuntimeError):
    """A Vercel API call failed; carries the status + parsed body for the reply."""

    def __init__(self, message: str, *, status: int = 0, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


# --------------------------------------------------------------------------- #
# Naming: the space <-> sandbox binding
# --------------------------------------------------------------------------- #

def sandbox_name(space_id: str) -> str:
    """The sandbox name bound to a Decillion space.

    Caspar space ids (``12@global``, ``local_<uuid>``) are not URL-safe and
    Vercel restricts names to ``[a-zA-Z0-9_-]{1,128}``. Sanitising alone can
    collide (``a@global`` and ``a-global`` both fold to ``a-global``), so the
    name carries a short digest of the *raw* id. Deterministic, so every
    creature derives the same name from the same space with no shared state.
    """
    raw = str(space_id or "").strip()
    if not raw:
        raise SandboxError("space_id is required to address a space sandbox")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    slug = _UNSAFE.sub("-", raw).strip("-").lower()[:80] or "space"
    return f"{NAME_PREFIX}-{slug}-{digest}"[:128]


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #

def _token() -> str:
    for env in ("VERCEL_TOKEN", "VERCEL_API_TOKEN", "VERCEL_ACCESS_TOKEN"):
        val = os.environ.get(env, "").strip()
        if val:
            return val
    raise SandboxError(
        "no Vercel API token in the creature environment — set VERCEL_TOKEN on the "
        "vercel_sandbox creature image (see scripts/deploy_sandbox_tool.py)")


def _scope_params() -> Dict[str, str]:
    params: Dict[str, str] = {}
    team = os.environ.get("VERCEL_TEAM_ID", "").strip()
    if team:
        params["teamId"] = team
    return params


def _project_id() -> str:
    return os.environ.get("VERCEL_PROJECT_ID", "").strip()


def _request(method: str, path: str, *, params: Optional[Dict[str, Any]] = None,
             json_body: Any = None, data: Optional[bytes] = None,
             headers: Optional[Dict[str, str]] = None, stream: bool = False,
             timeout: Optional[float] = None) -> requests.Response:
    hdrs = {"Authorization": f"Bearer {_token()}"}
    if headers:
        hdrs.update(headers)
    query = _scope_params()
    query.update({k: v for k, v in (params or {}).items() if v not in (None, "")})
    return requests.request(
        method, f"{API_BASE}{path}", params=query, json=json_body, data=data,
        headers=hdrs, stream=stream, timeout=timeout or HTTP_TIMEOUT)


def _json_body(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return (resp.text or "")[:500]


def _check(resp: requests.Response, what: str) -> Any:
    """Raise :class:`SandboxError` on a non-2xx, else return the parsed body."""
    if resp.status_code >= 300:
        body = _json_body(resp)
        message = body.get("error", {}).get("message") if isinstance(body, dict) else None
        raise SandboxError(f"{what} failed ({resp.status_code}): {message or body}",
                           status=resp.status_code, body=body)
    return _json_body(resp)


# --------------------------------------------------------------------------- #
# Sandbox lifecycle
# --------------------------------------------------------------------------- #

def _create_body(space_id: str, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    vcpus = int(payload.get("vcpus") or DEFAULT_VCPUS)
    body: Dict[str, Any] = {
        "name": name,
        "runtime": payload.get("runtime") or DEFAULT_RUNTIME,
        "timeout": int(payload.get("timeout_ms") or DEFAULT_TIMEOUT_MS),
        # Persistent = auto-snapshot on shutdown, so the space's files survive a
        # stopped VM and come back on the next `resume`.
        "persistent": True,
        "resources": {"vcpus": vcpus, "memory": int(payload.get("memory_mb") or vcpus * 2048)},
        # Tags are how an operator finds the sandbox belonging to a space in the
        # Vercel dashboard; the id is also recoverable from the name's digest.
        "tags": {"origin": "decillion", "spaceId": str(space_id)[:256]},
    }
    project = payload.get("project_id") or _project_id()
    if project:
        body["projectId"] = project
    for key, field in (("ports", "ports"), ("env", "env"), ("source", "source"),
                       ("network_policy", "networkPolicy"), ("mounts", "mounts")):
        if payload.get(key) is not None:
            body[field] = payload[key]
    return body


def _get_sandbox(name: str, *, resume: bool) -> Dict[str, Any]:
    resp = _request("GET", f"/v2/sandboxes/{name}",
                    params={"resume": "true" if resume else "false", "projectId": _project_id()})
    return _check(resp, f"get sandbox {name}")


def _create(space_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create the space's sandbox, or adopt the existing one.

    Space creation is retried by Nest and can race with a manual provision, so a
    name that already exists is a success, not an error: we resume it instead.
    """
    name = sandbox_name(space_id)
    resp = _request("POST", "/v2/sandboxes", json_body=_create_body(space_id, name, payload))
    if resp.status_code == 409:
        created = _get_sandbox(name, resume=True)
        created["adopted"] = True
        return created
    created = _check(resp, f"create sandbox {name}")
    created["adopted"] = False
    return created


def _session(space_id: str, *, auto_create: bool = True,
             create_payload: Optional[Dict[str, Any]] = None) -> str:
    """Resolve (and if needed resume or create) the space's live session id."""
    name = sandbox_name(space_id)
    cached = _SESSIONS.get(name)
    if cached and cached[1] > time.monotonic():
        return cached[0]
    try:
        info = _get_sandbox(name, resume=True)
    except SandboxError as exc:
        if exc.status not in (404, 410) or not auto_create:
            raise
        info = _create(space_id, create_payload or {})
    session_id = ((info.get("session") or {}).get("id") or ""
                  or (info.get("sandbox") or {}).get("currentSessionId") or "")
    if not session_id:
        raise SandboxError(f"sandbox {name} has no running session", body=info)
    _SESSIONS[name] = (session_id, time.monotonic() + _SESSION_TTL)
    return session_id


def _forget_session(space_id: str) -> None:
    _SESSIONS.pop(sandbox_name(space_id), None)


def _with_session(space_id: str, call) -> Any:
    """Run ``call(session_id)``, re-resolving once if the session went away.

    A named sandbox is stopped after its timeout and its session id dies with
    it; the next call must resume from the snapshot rather than fail. Every
    session-scoped operation goes through here so that recovery is automatic.
    """
    session_id = _session(space_id)
    try:
        return call(session_id)
    except SandboxError as exc:
        if exc.status not in (404, 409, 410):
            raise
        _forget_session(space_id)
        return call(_session(space_id))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n… [truncated, {len(text) - MAX_OUTPUT_CHARS} more chars]"


def _command_spec(payload: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Split the caller's request into Vercel's ``command`` + ``args``.

    Agents write shell lines (``pip install -r req.txt && pytest -q``), but the
    API takes a program plus argv. Unless the caller passes an explicit ``args``
    list we therefore hand the whole line to ``sh -c``, which is what makes
    pipes, redirects and ``&&`` work at all.
    """
    command = str(payload.get("command") or payload.get("cmd") or payload.get("task") or "").strip()
    if not command:
        raise SandboxError("command is required")
    args = payload.get("args")
    if isinstance(args, list):
        return command, [str(a) for a in args]
    if payload.get("shell") is False:
        parts = command.split()
        return parts[0], parts[1:]
    return "sh", ["-c", command]


def _drain_ndjson(resp: requests.Response) -> Tuple[Dict[str, Any], str, str]:
    """Consume the exec ND-JSON stream into (command record, stdout, stderr)."""
    record: Dict[str, Any] = {}
    out: List[str] = []
    err: List[str] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        if isinstance(event.get("command"), dict):
            record = event["command"]
        stream, data = event.get("stream"), event.get("data")
        if not isinstance(data, str):
            continue
        if stream == "stderr":
            err.append(data)
        elif stream == "stdout":
            out.append(data)
    return record, "".join(out), "".join(err)


def _exec(space_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    command, args = _command_spec(payload)
    body: Dict[str, Any] = {"command": command, "args": args, "wait": True, "logs": True,
                            "timeout": int(payload.get("timeout_ms") or EXEC_TIMEOUT_MS)}
    if payload.get("cwd"):
        body["cwd"] = str(payload["cwd"])
    if isinstance(payload.get("env"), dict):
        body["env"] = {str(k): str(v) for k, v in payload["env"].items()}
    if payload.get("sudo"):
        body["sudo"] = True

    started = time.time()

    def run(session_id: str) -> Tuple[Dict[str, Any], str, str, str]:
        # The read timeout must outlive the command itself: with wait+logs the
        # response streams for the whole run.
        read_timeout = body["timeout"] / 1000.0 + HTTP_TIMEOUT
        resp = _request("POST", f"/v2/sandboxes/sessions/{session_id}/cmd",
                        json_body=body, stream=True, timeout=read_timeout)
        if resp.status_code >= 300:
            _check(resp, "exec")
        record, out, err = _drain_ndjson(resp)
        return record, out, err, session_id

    record, out, err, session_id = _with_session(space_id, run)
    exit_code = record.get("exitCode")
    return {
        "ok": exit_code == 0,
        "action": "exec",
        "space_id": space_id,
        "sandbox": sandbox_name(space_id),
        "session_id": session_id,
        "command_id": record.get("id"),
        "exit_code": exit_code,
        "stdout": _truncate(out),
        "stderr": _truncate(err),
        "duration_ms": record.get("durationMs") or int((time.time() - started) * 1000),
    }


# --------------------------------------------------------------------------- #
# Filesystem
# --------------------------------------------------------------------------- #

def _file_entries(payload: Dict[str, Any]) -> List[Tuple[str, bytes]]:
    files = payload.get("files")
    if not isinstance(files, list):
        path = payload.get("path") or payload.get("file_name") or payload.get("target_path")
        if not path:
            raise SandboxError("write needs `path` + `content`, or a `files` list")
        files = [{"path": path, "content": payload.get("content", ""),
                  "encoding": payload.get("encoding", "text")}]
    entries: List[Tuple[str, bytes]] = []
    for spec in files:
        if not isinstance(spec, dict) or not spec.get("path"):
            raise SandboxError("each entry of `files` needs a `path`")
        content = spec.get("content", "")
        if str(spec.get("encoding") or "text").lower() == "base64":
            data = base64.b64decode(content or "")
        else:
            data = str(content).encode("utf-8")
        entries.append((str(spec["path"]), data))
    return entries


def _tarball(entries: List[Tuple[str, bytes]], cwd: Optional[str]) -> Tuple[bytes, Optional[str]]:
    """Pack files into the gzipped tar the write endpoint extracts.

    Paths are extracted *relative to* ``x-cwd`` (the home dir when unset), so an
    absolute target is expressed as "extract into ``/``" plus a root-relative
    arcname. Mixing the two in one call has no single correct answer, so it is
    rejected instead of silently writing to the wrong place.
    """
    absolute = [p for p, _ in entries if p.startswith("/")]
    if cwd is None and absolute:
        if len(absolute) != len(entries):
            raise SandboxError("mixed absolute and relative paths — pass an explicit `cwd`")
        cwd = "/"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, data in entries:
            info = tarfile.TarInfo(name=path.lstrip("/"))
            info.size = len(data)
            info.mtime = int(time.time())
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue(), cwd


def _write(space_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    entries = _file_entries(payload)
    blob, cwd = _tarball(entries, payload.get("cwd"))
    headers = {"Content-Type": "application/gzip"}
    if cwd:
        headers["x-cwd"] = cwd

    def run(session_id: str) -> str:
        resp = _request("POST", f"/v2/sandboxes/sessions/{session_id}/fs/write",
                        data=blob, headers=headers)
        _check(resp, "write files")
        return session_id

    session_id = _with_session(space_id, run)
    return {"ok": True, "action": "write", "space_id": space_id,
            "sandbox": sandbox_name(space_id), "session_id": session_id,
            "cwd": cwd, "written": [p for p, _ in entries],
            "bytes": sum(len(d) for _, d in entries)}


def _read(space_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    path = str(payload.get("path") or "").strip()
    if not path:
        raise SandboxError("path is required")
    body = {"path": path}
    if payload.get("cwd"):
        body["cwd"] = str(payload["cwd"])

    def run(session_id: str) -> Tuple[bytes, str]:
        resp = _request("POST", f"/v2/sandboxes/sessions/{session_id}/fs/read", json_body=body)
        if resp.status_code >= 300:
            _check(resp, f"read {path}")
        return resp.content[:MAX_READ_BYTES], session_id

    data, session_id = _with_session(space_id, run)
    result = {"ok": True, "action": "read", "space_id": space_id,
              "sandbox": sandbox_name(space_id), "session_id": session_id,
              "path": path, "bytes": len(data)}
    try:
        result["content"] = _truncate(data.decode("utf-8"))
        result["encoding"] = "text"
    except UnicodeDecodeError:
        result["content"] = base64.b64encode(data).decode("ascii")
        result["encoding"] = "base64"
    return result


def _mkdir(space_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    path = str(payload.get("path") or "").strip()
    if not path:
        raise SandboxError("path is required")
    body = {"path": path}
    if payload.get("cwd"):
        body["cwd"] = str(payload["cwd"])

    def run(session_id: str) -> str:
        resp = _request("POST", f"/v2/sandboxes/sessions/{session_id}/fs/mkdir", json_body=body)
        _check(resp, f"mkdir {path}")
        return session_id

    session_id = _with_session(space_id, run)
    return {"ok": True, "action": "mkdir", "space_id": space_id, "path": path,
            "sandbox": sandbox_name(space_id), "session_id": session_id}


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #

def _summarize(space_id: str, info: Dict[str, Any], action: str) -> Dict[str, Any]:
    sandbox = info.get("sandbox") or {}
    session = info.get("session") or {}
    return {
        "ok": True,
        "action": action,
        "space_id": space_id,
        "sandbox": sandbox.get("name") or sandbox_name(space_id),
        "status": sandbox.get("status") or session.get("status"),
        "session_id": session.get("id") or sandbox.get("currentSessionId"),
        "runtime": sandbox.get("runtime") or session.get("runtime"),
        "region": sandbox.get("region") or session.get("region"),
        "cwd": sandbox.get("cwd") or session.get("cwd"),
        "expires_at": sandbox.get("expiresAt"),
        # Public URLs of any exposed ports — how an agent hands a running dev
        # server or preview back to the humans in the space.
        "routes": [{"url": r.get("url"), "port": r.get("port")}
                   for r in (info.get("routes") or []) if isinstance(r, dict)],
        "adopted": info.get("adopted"),
        "resumed": info.get("resumed"),
    }


def _action_create(space_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    _forget_session(space_id)
    return _summarize(space_id, _create(space_id, payload), "create")


def _action_info(space_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resume = bool(payload.get("resume", False))
    try:
        return _summarize(space_id, _get_sandbox(sandbox_name(space_id), resume=resume), "info")
    except SandboxError as exc:
        if exc.status in (404, 410):
            return {"ok": False, "action": "info", "space_id": space_id,
                    "sandbox": sandbox_name(space_id), "exists": False,
                    "error": "no sandbox is bound to this space"}
        raise


def _action_start(space_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Resume the space's sandbox (creating it if it was never provisioned)."""
    _forget_session(space_id)
    try:
        info = _get_sandbox(sandbox_name(space_id), resume=True)
    except SandboxError as exc:
        if exc.status not in (404, 410):
            raise
        info = _create(space_id, payload)
    return _summarize(space_id, info, "start")


def _action_stop(space_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stop the running session, keeping the snapshot. Never resumes on the way
    in — asking a stopped sandbox to stop must not boot a VM to shut it down."""
    name = sandbox_name(space_id)
    try:
        info = _get_sandbox(name, resume=False)
    except SandboxError as exc:
        if exc.status in (404, 410):
            return {"ok": True, "action": "stop", "space_id": space_id,
                    "sandbox": name, "already_stopped": True, "exists": False}
        raise
    sandbox = info.get("sandbox") or {}
    session_id = ((info.get("session") or {}).get("id") or sandbox.get("currentSessionId") or "")
    if sandbox.get("status") != "running" or not session_id:
        _forget_session(space_id)
        return {"ok": True, "action": "stop", "space_id": space_id, "sandbox": name,
                "already_stopped": True, "status": sandbox.get("status")}
    resp = _request("POST", f"/v2/sandboxes/sessions/{session_id}/stop")
    if resp.status_code not in (404, 410):
        _check(resp, "stop session")
    _forget_session(space_id)
    return {"ok": True, "action": "stop", "space_id": space_id, "sandbox": name,
            "session_id": session_id,
            "note": "filesystem snapshotted; the next command resumes it"}


def _action_delete(space_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Destroy the space's sandbox. Idempotent — an already-gone sandbox is a
    success, because this runs on space deletion and must not leave Nest with a
    binding it cannot clear."""
    name = sandbox_name(space_id)
    resp = _request("DELETE", f"/v2/sandboxes/{name}", params={"projectId": _project_id()})
    _forget_session(space_id)
    if resp.status_code in (404, 410):
        return {"ok": True, "action": "delete", "space_id": space_id, "sandbox": name,
                "deleted": False, "already_absent": True}
    body = _check(resp, f"delete sandbox {name}")
    return {"ok": True, "action": "delete", "space_id": space_id, "sandbox": name,
            "deleted": True, "status": (body.get("sandbox") or {}).get("status")}


def _action_list(space_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = _check(_request("GET", "/v2/sandboxes", params={"projectId": _project_id()}),
                  "list sandboxes")
    rows = body.get("sandboxes") if isinstance(body, dict) else None
    return {"ok": True, "action": "list",
            "sandboxes": [{"name": r.get("name"), "status": r.get("status"),
                           "spaceId": (r.get("tags") or {}).get("spaceId")}
                          for r in (rows or []) if isinstance(r, dict)]}


_ACTIONS = {
    "create": _action_create,
    "provision": _action_create,
    "start": _action_start,
    "resume": _action_start,
    "info": _action_info,
    "status": _action_info,
    "exec": _exec,
    "run": _exec,
    "shell": _exec,
    "write": _write,
    "read": _read,
    "mkdir": _mkdir,
    "stop": _action_stop,
    "suspend": _action_stop,
    "delete": _action_delete,
    "destroy": _action_delete,
    "remove": _action_delete,
    "list": _action_list,
}

# Actions that operate on the whole account rather than one space.
_SPACELESS = {"list"}


def _normalize_action(function_name: str, payload: Dict[str, Any]) -> str:
    """Pick the action from the signal's ``function`` or the payload.

    Davinci's bridge executor sends the catalog's routing function, but a lead
    agent reaching the tool through the generic ``invoke`` route names the
    action in its arguments instead — both must work.
    """
    for candidate in (payload.get("action"), payload.get("function"), function_name):
        if isinstance(candidate, str) and candidate.strip() and candidate.strip() != "invoke":
            return candidate.strip().lower()
    return "exec" if payload.get("command") else "info"


def _space_id(payload: Dict[str, Any]) -> str:
    for key in ("space_id", "spaceId", "store_id", "storeId"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    raise SandboxError("space_id is required — the sandbox is bound to a Decillion space")


def invoke(function_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(payload or {})
    action = _normalize_action(function_name, payload)
    handler = _ACTIONS.get(action)
    if handler is None:
        return {"ok": False, "error": f"unknown action '{action}'",
                "actions": sorted(set(_ACTIONS))}
    try:
        space_id = "" if action in _SPACELESS else _space_id(payload)
        return handler(space_id, payload)
    except SandboxError as exc:
        return {"ok": False, "action": action, "error": str(exc),
                "status": exc.status or None}
    except requests.RequestException as exc:
        return {"ok": False, "action": action,
                "error": f"vercel api unreachable: {exc}"}
