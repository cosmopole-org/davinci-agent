"""Tests for the vercel_sandbox tool creature — the per-space cloud sandbox.

The tool is driven against a fake ``requests`` module that records every HTTP
call and returns Vercel-shaped responses, so the real tool.py logic (name
binding, session resume, ND-JSON exec parsing, tarball writes, idempotent
delete) is exercised with no network and no Vercel account.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import os

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO / "tools" / "vercel_sandbox" / "tool.py"

SPACE = "42@global"

_ENV_KEYS = ("VERCEL_TOKEN", "VERCEL_API_TOKEN", "VERCEL_ACCESS_TOKEN", "VERCEL_TEAM_ID",
             "VERCEL_PROJECT_ID", "VERCEL_SANDBOX_MAX_OUTPUT", "VERCEL_SANDBOX_PREFIX")


@pytest.fixture(autouse=True)
def _isolate_module_globals():
    """The loader swaps `requests` and sets Vercel env vars — undo both after
    each test so no other test in the session inherits the fakes."""
    saved_requests = sys.modules.get("requests")
    saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
    yield
    if saved_requests is None:
        sys.modules.pop("requests", None)
    else:
        sys.modules["requests"] = saved_requests
    for key, val in saved_env.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class _Resp:
    def __init__(self, status: int = 200, body: Any = None, *, content: bytes = b"",
                 lines: Optional[List[Dict[str, Any]]] = None) -> None:
        self.status_code = status
        self._body = body
        self.content = content
        self.text = "" if body is None else json.dumps(body)
        self._lines = lines or []

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield json.dumps(line)


class _FakeRequests:
    """Stands in for the ``requests`` module inside tool.py."""

    class RequestException(Exception):
        pass

    def __init__(self, handler: Callable[[str, str, Dict[str, Any]], _Resp]) -> None:
        self.handler = handler
        self.calls: List[Dict[str, Any]] = []

    def request(self, method: str, url: str, **kw: Any) -> _Resp:
        self.calls.append({"method": method, "url": url, **kw})
        return self.handler(method, url, kw)

    def paths(self) -> List[str]:
        return [f"{c['method']} {c['url'].split('api.vercel.com')[-1].split('?')[0]}"
                for c in self.calls]

    def call(self, method: str, needle: str) -> Dict[str, Any]:
        return next(c for c in self.calls if c["method"] == method and needle in c["url"])


def _load(handler: Callable[[str, str, Dict[str, Any]], _Resp], **env: str):
    fake = _FakeRequests(handler)
    sys.modules["requests"] = fake  # type: ignore[assignment]
    for key, val in {"VERCEL_TOKEN": "tok_test", "VERCEL_TEAM_ID": "team_1",
                     "VERCEL_PROJECT_ID": "prj_1", **env}.items():
        os.environ[key] = val
    spec = importlib.util.spec_from_file_location("vercel_sandbox_tool_" + str(id(handler)),
                                                  _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, fake


def _sandbox_body(name: str, status: str = "running", session: str = "sess_1") -> Dict[str, Any]:
    return {"sandbox": {"name": name, "status": status, "currentSessionId": session,
                        "runtime": "node24", "cwd": "/vercel/sandbox"},
            "session": {"id": session, "status": status, "cwd": "/vercel/sandbox"},
            "routes": [{"url": "https://x.vercel.run", "port": 3000}]}


def _ok_handler(**overrides: Callable[[str, str, Dict[str, Any]], Optional[_Resp]]):
    """A handler that answers every endpoint with a healthy response."""

    def handler(method: str, url: str, kw: Dict[str, Any]) -> _Resp:
        for key, fn in overrides.items():
            resp = fn(method, url, kw)
            if resp is not None:
                return resp
        name = url.rstrip("/").split("/")[-1]
        if method == "POST" and url.endswith("/v2/sandboxes"):
            return _Resp(200, _sandbox_body((kw.get("json") or {}).get("name", "")))
        if method == "GET" and "/v2/sandboxes/" in url and "/sessions/" not in url:
            return _Resp(200, _sandbox_body(name))
        if method == "DELETE":
            return _Resp(200, {"sandbox": {"name": name, "status": "stopped"}})
        if url.endswith("/cmd"):
            return _Resp(200, lines=[
                {"stream": "stdout", "data": "hello\n"},
                {"stream": "stderr", "data": "warn\n"},
                {"command": {"id": "cmd_1", "exitCode": 0, "durationMs": 12}},
            ])
        if url.endswith("/fs/read"):
            return _Resp(200, content=b"file body")
        return _Resp(200, {})

    return handler


# --------------------------------------------------------------------------- #
# The space <-> sandbox binding
# --------------------------------------------------------------------------- #

def test_sandbox_name_is_deterministic_url_safe_and_collision_free():
    mod, _ = _load(_ok_handler())
    name = mod.sandbox_name(SPACE)
    assert name == mod.sandbox_name(SPACE)          # deterministic
    assert name.replace("-", "").replace("_", "").isalnum()  # url-safe
    assert len(name) <= 128
    # Two space ids that sanitise to the same slug still get distinct names.
    assert mod.sandbox_name("a@global") != mod.sandbox_name("a-global")


def test_every_action_but_list_requires_a_space_id():
    mod, fake = _load(_ok_handler())
    for function in ("exec", "write", "read", "info", "delete"):
        out = mod.invoke(function, {"command": "ls", "path": "x", "content": "y"})
        assert out["ok"] is False and "space_id" in out["error"]
    assert fake.calls == []  # nothing reached the API


# --------------------------------------------------------------------------- #
# create / delete — the space lifecycle hooks Nest signals
# --------------------------------------------------------------------------- #

def test_create_names_the_sandbox_after_the_space_and_tags_it():
    mod, fake = _load(_ok_handler())
    out = mod.invoke("create", {"space_id": SPACE, "ports": [3000]})

    body = fake.call("POST", "/v2/sandboxes")["json"]
    assert body["name"] == mod.sandbox_name(SPACE)
    assert body["tags"] == {"origin": "decillion", "spaceId": SPACE}
    assert body["persistent"] is True          # the space's files survive a stop
    assert body["ports"] == [3000]
    assert body["projectId"] == "prj_1"
    assert fake.calls[0]["params"]["teamId"] == "team_1"
    assert out["ok"] is True and out["routes"] == [{"url": "https://x.vercel.run", "port": 3000}]


def test_create_adopts_an_existing_sandbox_instead_of_failing():
    def conflict(method, url, kw):
        if method == "POST" and url.endswith("/v2/sandboxes"):
            return _Resp(409, {"error": {"message": "already exists"}})
        return None

    mod, fake = _load(_ok_handler(conflict=conflict))
    out = mod.invoke("create", {"space_id": SPACE})

    assert out["ok"] is True and out["adopted"] is True
    assert fake.paths()[-1].startswith("GET")
    assert fake.calls[-1]["params"]["resume"] == "true"


def test_delete_targets_the_space_sandbox_and_is_idempotent():
    mod, fake = _load(_ok_handler())
    out = mod.invoke("delete", {"space_id": SPACE})
    assert out["ok"] is True and out["deleted"] is True
    assert fake.calls[0]["method"] == "DELETE"
    assert mod.sandbox_name(SPACE) in fake.calls[0]["url"]

    # An already-destroyed sandbox must still report success, or Nest can never
    # clear the binding for a space the user deleted.
    gone = _load(_ok_handler(missing=lambda m, u, k: _Resp(404, {"error": {"message": "nope"}})
                             if m == "DELETE" else None))
    out = gone[0].invoke("delete", {"space_id": SPACE})
    assert out["ok"] is True and out["already_absent"] is True


def test_delete_reports_a_real_api_failure():
    mod, _ = _load(_ok_handler(boom=lambda m, u, k: _Resp(500, {"error": {"message": "kaboom"}})
                               if m == "DELETE" else None))
    out = mod.invoke("delete", {"space_id": SPACE})
    assert out["ok"] is False and "kaboom" in out["error"]


# --------------------------------------------------------------------------- #
# exec — what the agents actually call
# --------------------------------------------------------------------------- #

def test_exec_runs_the_line_through_a_shell_and_returns_streams():
    mod, fake = _load(_ok_handler())
    out = mod.invoke("exec", {"space_id": SPACE, "command": "echo hello && ls | wc -l"})

    body = fake.call("POST", "/cmd")["json"]
    assert body["command"] == "sh" and body["args"] == ["-c", "echo hello && ls | wc -l"]
    assert body["wait"] is True and body["logs"] is True
    assert out["ok"] is True and out["exit_code"] == 0
    assert out["stdout"] == "hello\n" and out["stderr"] == "warn\n"
    assert out["sandbox"] == mod.sandbox_name(SPACE) and out["session_id"] == "sess_1"


def test_exec_honours_an_explicit_argv():
    mod, fake = _load(_ok_handler())
    mod.invoke("exec", {"space_id": SPACE, "command": "python3", "args": ["-c", "print(1)"]})
    body = fake.call("POST", "/cmd")["json"]
    assert body["command"] == "python3" and body["args"] == ["-c", "print(1)"]


def test_exec_reports_a_nonzero_exit_without_claiming_success():
    def failing(method, url, kw):
        if url.endswith("/cmd"):
            return _Resp(200, lines=[{"stream": "stderr", "data": "boom"},
                                     {"command": {"id": "c", "exitCode": 2}}])
        return None

    mod, _ = _load(_ok_handler(failing=failing))
    out = mod.invoke("exec", {"space_id": SPACE, "command": "false"})
    assert out["ok"] is False and out["exit_code"] == 2 and out["stderr"] == "boom"


def test_exec_resumes_a_stopped_session_and_retries_once():
    state = {"cmd_calls": 0}

    def flaky(method, url, kw):
        if url.endswith("/cmd"):
            state["cmd_calls"] += 1
            if state["cmd_calls"] == 1:
                return _Resp(410, {"error": {"message": "session gone"}})
        return None

    mod, fake = _load(_ok_handler(flaky=flaky))
    out = mod.invoke("exec", {"space_id": SPACE, "command": "ls"})

    assert out["ok"] is True and state["cmd_calls"] == 2
    # The retry re-resolved the session through a resuming GET.
    assert [c for c in fake.calls if c["method"] == "GET"][-1]["params"]["resume"] == "true"


def test_exec_creates_the_sandbox_when_the_space_never_got_one():
    def missing_then_created(method, url, kw):
        if method == "GET" and "/v2/sandboxes/" in url:
            return _Resp(404, {"error": {"message": "not found"}})
        return None

    mod, fake = _load(_ok_handler(missing=missing_then_created))
    out = mod.invoke("exec", {"space_id": SPACE, "command": "ls"})
    assert out["ok"] is True
    assert "POST /v2/sandboxes" in fake.paths()


def test_exec_output_is_capped_for_the_model_context():
    def flood(method, url, kw):
        if url.endswith("/cmd"):
            return _Resp(200, lines=[{"stream": "stdout", "data": "x" * 200_000},
                                     {"command": {"id": "c", "exitCode": 0}}])
        return None

    mod, _ = _load(_ok_handler(flood=flood), VERCEL_SANDBOX_MAX_OUTPUT="1000")
    out = mod.invoke("exec", {"space_id": SPACE, "command": "yes"})
    assert len(out["stdout"]) < 1200 and "truncated" in out["stdout"]


# --------------------------------------------------------------------------- #
# filesystem
# --------------------------------------------------------------------------- #

def _extract(blob: bytes) -> Dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}


def test_write_uploads_a_gzipped_tarball_relative_to_the_extract_dir():
    mod, fake = _load(_ok_handler())
    out = mod.invoke("write", {"space_id": SPACE, "path": "/tmp/app/main.py",
                               "content": "print('hi')"})

    call = fake.call("POST", "/fs/write")
    assert call["headers"]["Content-Type"] == "application/gzip"
    assert call["headers"]["x-cwd"] == "/"
    assert _extract(call["data"]) == {"tmp/app/main.py": b"print('hi')"}
    assert out["ok"] is True and out["written"] == ["/tmp/app/main.py"]


def test_write_supports_multiple_files_and_base64_content():
    mod, fake = _load(_ok_handler())
    mod.invoke("write", {"space_id": SPACE, "files": [
        {"path": "a.txt", "content": "alpha"},
        {"path": "b.bin", "content": "AAEC", "encoding": "base64"},
    ]})
    files = _extract(fake.call("POST", "/fs/write")["data"])
    assert files == {"a.txt": b"alpha", "b.bin": b"\x00\x01\x02"}


def test_write_rejects_mixed_absolute_and_relative_paths():
    mod, fake = _load(_ok_handler())
    out = mod.invoke("write", {"space_id": SPACE, "files": [{"path": "/etc/a"},
                                                            {"path": "b"}]})
    assert out["ok"] is False and "cwd" in out["error"]
    assert not [c for c in fake.calls if "fs/write" in c["url"]]


def test_read_returns_text_and_falls_back_to_base64():
    mod, fake = _load(_ok_handler())
    out = mod.invoke("read", {"space_id": SPACE, "path": "notes.md"})
    assert out["content"] == "file body" and out["encoding"] == "text"
    assert fake.call("POST", "/fs/read")["json"] == {"path": "notes.md"}

    binary = _load(_ok_handler(bin=lambda m, u, k: _Resp(200, content=b"\xff\xfe")
                               if u.endswith("/fs/read") else None))[0]
    out = binary.invoke("read", {"space_id": SPACE, "path": "a.png"})
    assert out["encoding"] == "base64" and out["content"] == "//4="


# --------------------------------------------------------------------------- #
# lifecycle helpers
# --------------------------------------------------------------------------- #

def test_stop_never_resumes_a_stopped_sandbox_just_to_stop_it():
    stopped = _load(_ok_handler(
        s=lambda m, u, k: _Resp(200, _sandbox_body("n", status="stopped"))
        if m == "GET" and "/sessions/" not in u else None))
    mod, fake = stopped
    out = mod.invoke("stop", {"space_id": SPACE})

    assert out["ok"] is True and out["already_stopped"] is True
    assert fake.calls[0]["params"]["resume"] == "false"
    assert not [c for c in fake.calls if c["url"].endswith("/stop")]


def test_stop_stops_a_running_session_and_keeps_the_snapshot():
    mod, fake = _load(_ok_handler())
    out = mod.invoke("stop", {"space_id": SPACE})
    assert out["ok"] is True and fake.paths()[-1].endswith("/stop")
    assert "resumes" in out["note"]


def test_info_on_an_unprovisioned_space_says_so_without_creating_one():
    mod, fake = _load(_ok_handler(missing=lambda m, u, k: _Resp(404, {}) if m == "GET" else None))
    out = mod.invoke("info", {"space_id": SPACE})
    assert out["ok"] is False and out["exists"] is False
    assert "POST /v2/sandboxes" not in fake.paths()


def test_missing_token_is_reported_rather_than_calling_the_api():
    mod, fake = _load(_ok_handler())
    for key in ("VERCEL_TOKEN", "VERCEL_API_TOKEN", "VERCEL_ACCESS_TOKEN"):
        os.environ.pop(key, None)
    try:
        out = mod.invoke("exec", {"space_id": SPACE, "command": "ls"})
    finally:
        os.environ["VERCEL_TOKEN"] = "tok_test"
    assert out["ok"] is False and "VERCEL_TOKEN" in out["error"]
    assert fake.calls == []


def test_deploy_bakes_every_token_spelling_the_tool_accepts():
    """The tool accepts three names for the API token and the CI gate lets any
    of them enable the deploy — so the bake list must carry all three, or an
    operator who set one of the aliases deploys a creature that looks healthy
    and refuses every call."""
    sys.path.insert(0, str(_REPO / "scripts"))
    import deploy_and_test  # noqa: PLC0415

    for name in ("VERCEL_TOKEN", "VERCEL_API_TOKEN", "VERCEL_ACCESS_TOKEN"):
        for key in ("VERCEL_TOKEN", "VERCEL_API_TOKEN", "VERCEL_ACCESS_TOKEN"):
            os.environ.pop(key, None)
        os.environ[name] = "tok_from_" + name
        assert deploy_and_test.sandbox_bake_env().get(name) == "tok_from_" + name

    # And every tuning knob tool.py reads is bakeable, so a setting an operator
    # exports actually reaches the creature.
    tool_src = (_REPO / "tools" / "vercel_sandbox" / "tool.py").read_text()
    read_by_tool = set(re.findall(r'environ\.get\("(VERCEL_[A-Z_]+)"', tool_src))
    read_by_tool |= set(re.findall(r'"(VERCEL_[A-Z_]+)"', tool_src)) & {
        "VERCEL_TOKEN", "VERCEL_API_TOKEN", "VERCEL_ACCESS_TOKEN"}
    assert read_by_tool <= set(deploy_and_test.SANDBOX_ENV_KEYS), (
        f"not baked: {sorted(read_by_tool - set(deploy_and_test.SANDBOX_ENV_KEYS))}")


def test_unknown_function_lists_the_supported_ones():
    mod, _ = _load(_ok_handler())
    out = mod.invoke("teleport", {"space_id": SPACE})
    assert out["ok"] is False and "exec" in out["actions"]
