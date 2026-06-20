"""python_exec tool creature — execute Python code in an isolated subprocess.

Real implementation:

* Runs the submitted code in a **separate Python subprocess** so that a wall
  clock timeout, CPU/address-space resource limits, and a scratch working
  directory can be enforced independently of the runtime process. (The whole
  creature already runs inside a gVisor sandbox on the Caspar node; this adds a
  second, in-process layer of bounding.)
* Captures ``stdout`` / ``stderr`` and the value bound to a ``result`` variable
  (JSON-encoded if possible) plus the names of top-level variables produced.
* Optional ``requirements`` are pip-installed before the run, and optional
  ``files`` are materialised into the working directory.
* ``pandas`` / ``numpy`` are available in the image for data work.

Signal payload (function ``execute`` or ``invoke``)::

    {
      "code": "result = sum(range(10))",      # required
      "stdin": "...",                          # optional, fed to the process
      "timeout": 30,                            # optional seconds (default 30)
      "files": {"data.csv": "a,b\\n1,2\\n"},   # optional files in workdir
      "requirements": ["requests"],            # optional pip installs
      "env": {"FOO": "bar"}                    # optional extra env vars
    }
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300
MAX_OUTPUT = 200_000  # bytes of captured stdout/stderr to return

# Code wrapper: runs the user's code, then serialises a ``result`` variable and
# the set of user-defined globals as a trailing JSON sentinel line on stdout.
_WRAPPER = textwrap.dedent(
    '''
    import json as _json, sys as _sys

    _SENTINEL = "__DAVINCI_PYEXEC_RESULT__"
    _user_globals = {{"__name__": "__main__"}}
    _exit_code = 0
    try:
        with open({code_path!r}, "r", encoding="utf-8") as _fh:
            _src = _fh.read()
        exec(compile(_src, "<davinci_python_exec>", "exec"), _user_globals)
    except SystemExit as _se:
        _exit_code = int(_se.code) if isinstance(_se.code, int) else 0
    except BaseException:
        import traceback as _tb
        _tb.print_exc()
        _exit_code = 1

    def _encode(_v):
        try:
            _json.dumps(_v)
            return _v
        except Exception:
            return repr(_v)

    _result = _user_globals.get("result", None)
    _vars = sorted(
        k for k, v in _user_globals.items()
        if not k.startswith("_") and not callable(v)
    )
    _sys.stdout.flush()
    print(_SENTINEL + _json.dumps({{"result": _encode(_result), "vars": _vars}}))
    _sys.exit(_exit_code)
    '''
)


def _set_rlimits():  # pragma: no cover - only runs in child process on POSIX
    """Bound the child's CPU time and address space as a defence in depth."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (90, 120))
        # Roomy enough for pandas/numpy on real-world data while still bounding a
        # runaway allocation. gVisor provides the hard isolation boundary around
        # the whole creature.
        soft = 2048 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def _run(code: str, *, stdin: str, timeout: int, files: dict, env_extra: dict) -> dict:
    workdir = tempfile.mkdtemp(prefix="pyexec_")
    code_path = os.path.join(workdir, "_snippet.py")
    runner_path = os.path.join(workdir, "_runner.py")
    with open(code_path, "w", encoding="utf-8") as fh:
        fh.write(code)
    with open(runner_path, "w", encoding="utf-8") as fh:
        fh.write(_WRAPPER.format(code_path=code_path))

    for name, content in (files or {}).items():
        safe = os.path.normpath(name).lstrip("/")
        dest = os.path.join(workdir, safe)
        os.makedirs(os.path.dirname(dest) or workdir, exist_ok=True)
        mode, data = ("wb", content) if isinstance(content, (bytes, bytearray)) else ("w", str(content))
        with open(dest, mode) as wf:
            wf.write(data)

    env = dict(os.environ)
    env.update({k: str(v) for k, v in (env_extra or {}).items()})
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        proc = subprocess.run(
            [sys.executable, runner_path],
            cwd=workdir, input=stdin, capture_output=True, text=True,
            timeout=timeout, env=env,
            preexec_fn=_set_rlimits if os.name == "posix" else None,
        )
    except subprocess.TimeoutExpired as exc:
        so = exc.stdout if isinstance(exc.stdout, str) else ""
        se = exc.stderr if isinstance(exc.stderr, str) else ""
        return {"ok": False, "error": f"execution timed out after {timeout}s",
                "stdout": so[:MAX_OUTPUT], "stderr": se[:MAX_OUTPUT], "timed_out": True}

    stdout = proc.stdout or ""
    result_obj, result_vars = None, []
    sentinel = "__DAVINCI_PYEXEC_RESULT__"
    if sentinel in stdout:
        head, _, tail = stdout.rpartition(sentinel)
        stdout = head
        try:
            meta = json.loads(tail.strip().splitlines()[0])
            result_obj, result_vars = meta.get("result"), meta.get("vars", [])
        except Exception:
            pass

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "result": result_obj,
        "vars": result_vars,
        "stdout": stdout[:MAX_OUTPUT],
        "stderr": (proc.stderr or "")[:MAX_OUTPUT],
        "workdir_files": sorted(os.listdir(workdir)),
    }


def _pip_install(requirements: list) -> dict:
    if not requirements:
        return {"installed": []}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "--disable-pip-version-check", *[str(r) for r in requirements]],
            capture_output=True, text=True, timeout=240,
        )
        return {"installed": requirements, "ok": proc.returncode == 0,
                "log": (proc.stdout + proc.stderr)[-2000:]}
    except Exception as exc:
        return {"installed": [], "ok": False, "error": str(exc)}


def invoke(function_name: str, payload: dict) -> dict:
    payload = payload or {}
    code = payload.get("code")
    if not code:
        task = payload.get("task")
        code = f"result = ({task})" if task else None
    if not code:
        return {"ok": False, "error": "no 'code' provided"}

    timeout = max(1, min(int(payload.get("timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT), MAX_TIMEOUT))
    out: dict = {}
    reqs = payload.get("requirements") or []
    if reqs:
        out["pip"] = _pip_install(reqs)

    run = _run(str(code), stdin=str(payload.get("stdin", "") or ""), timeout=timeout,
               files=payload.get("files") or {}, env_extra=payload.get("env") or {})
    out.update(run)
    out["function"] = function_name
    return out


if __name__ == "__main__":
    print(json.dumps(invoke("execute", {"code": "result = 6 * 7\nprint('hello')"}), indent=2))
