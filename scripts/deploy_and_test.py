#!/usr/bin/env python3
"""End-to-end deploy + test of Davinci as a Caspar docker creature swarm.

Pipeline (all over Caspar's signalling / action API):

  Phase 1  Deploy each Davinci tool as its own ``docker`` creature and wait for
           the node to build its image.
  Phase 2  Signal each tool creature directly (``/programs/runEntity``) and
           verify it returns a ``TOOL_RESPONSE`` in its VM logs.
  Phase 3  Deploy the Davinci agent itself as a ``docker`` creature (shipped as
           a tarball build context so the whole package lands in the image).
  Phase 4  Signal the Davinci creature with a task + a config naming the tool
           creatures. Davinci then signals those tool creatures *itself*
           through the node and aggregates their responses — proving creature-to
           -creature interaction over the Caspar signalling API.

Requires: a running Caspar node (default 127.0.0.1:8074), a reachable Docker
daemon with the ``runsc`` runtime + ``kasper`` network, and ``pycryptodome``.

Run:  python3 scripts/deploy_and_test.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from davinci.caspar_signaling import CasparSignalingClient, _log_text  # noqa: E402

NODE_HOST = os.environ.get("CASPAR_NODE_HOST", "127.0.0.1")
NODE_PORT = int(os.environ.get("CASPAR_NODE_PORT", "8074"))
# Address the *creature* uses to reach the node from inside the docker network.
NODE_HOST_FROM_VM = os.environ.get("CASPAR_NODE_HOST_FROM_VM", "172.17.0.1")
ADMIN_USER = "davinci_admin"

TOOLS_DIR = REPO / "tools"


def _discover_tools() -> list:
    """Build the tool-creature catalog from each tool's point.metadata.json.

    category/risk/requires_network drive Davinci's planner + permissions; the
    route function is taken from the metadata ``routes`` map (falling back to
    ``invoke``) so the live creature signals each tool with its real function.
    """
    tools = []
    for meta_path in sorted(TOOLS_DIR.glob("*/point.metadata.json")):
        meta = json.loads(meta_path.read_text())
        cats = meta.get("categories") or ["general"]
        routes = meta.get("routes") or {}
        function = routes.get(cats[0], "invoke")
        tools.append({
            "tool_id": meta["tool_id"],
            "category": cats[0],
            "risk": meta.get("risk_level", "medium"),
            "description": meta.get("description", meta["tool_id"]),
            "requires_network": bool(meta.get("requires_network", False)),
            "function": function,
        })
    return tools


# Tool creatures to deploy (every tool under tools/ with metadata).
TOOLS = _discover_tools()

# Heavier images (system packages / browser binaries) need a longer build window.
BUILD_TIMEOUT = {"browser_automation": 900, "sql_query": 480, "vector_search": 480,
                 "calendar_connector": 480}

GREEN, RED, CYAN, NC = "\033[0;32m", "\033[0;31m", "\033[0;36m", "\033[0m"


def info(m): print(f"{CYAN}[deploy]{NC} {m}", flush=True)
def ok(m):   print(f"{GREEN}[ ok ]{NC} {m}", flush=True)
def bad(m):  print(f"{RED}[fail]{NC} {m}", flush=True)


def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def b64_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode()


def docker_image_exists(program_id: str, entity_id: str) -> bool:
    image = f"{program_id.replace('@', '_')}/{entity_id}"
    try:
        out = subprocess.run(["docker", "images", "--format", "{{.Repository}}", image],
                             capture_output=True, text=True, timeout=15)
        return image in out.stdout
    except Exception:
        return False


def wait_for_image(program_id: str, entity_id: str, timeout: int = 240) -> bool:
    image = f"{program_id.replace('@', '_')}/{entity_id}"
    info(f"waiting for node to build image {image} (≤{timeout}s)…")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if docker_image_exists(program_id, entity_id):
            return True
        time.sleep(3)
    return False


# --------------------------------------------------------------------------- #
# Build contexts
# --------------------------------------------------------------------------- #

# Fallback Dockerfile for a tool that ships no Dockerfile of its own.
TOOL_DOCKERFILE = (
    "FROM public.ecr.aws/docker/library/python:3.12-slim\n"
    "WORKDIR /app\nCOPY . /app\n"
    # The node uploads the signal payload to /app/input before start; the dir
    # must exist or the upload (and thus container start) fails.
    "RUN mkdir -p /app/input\n"
    "ENV DAVINCI_INPUT_DIR=/app/input PYTHONUNBUFFERED=1\n"
    'CMD ["python", "tool_runtime.py"]\n'
)

DAVINCI_DOCKERFILE = (
    "FROM public.ecr.aws/docker/library/python:3.12-slim\n"
    "WORKDIR /app\n"
    "ADD bundle.tar /app\n"
    "RUN pip install --no-cache-dir --trusted-host pypi.org "
    "--trusted-host files.pythonhosted.org --trusted-host pypi.python.org pycryptodome\n"
    "RUN mkdir -p /app/input\n"
    "ENV DAVINCI_INPUT_DIR=/app/input PYTHONUNBUFFERED=1\n"
    'CMD ["python", "-m", "davinci.caspar_runtime"]\n'
)


def davinci_bundle_tar() -> bytes:
    """Tar the davinci package + orchestrator + runtime model into the build ctx."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel in ["caspar_orchestrator.py", "agentic_runtime.py"]:
            tar.add(REPO / rel, arcname=rel)
        for py in sorted((REPO / "davinci").glob("*.py")):
            tar.add(py, arcname=f"davinci/{py.name}")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #

def deploy_tool(c: CasparSignalingClient, tool: dict) -> dict:
    tid = tool["tool_id"]
    tool_dir = REPO / "tools" / tid
    machine_id = c.create_machine_creature(f"m-tool-{tid}")
    program_id = c.create_program(machine_id, f"/tools/{tid}", "docker", f"tool {tid}")

    # Build context: the shared runtime + the tool's real implementation, plus
    # its requirements.txt so the tool's own Dockerfile can install its deps.
    files = {"tool_runtime.py": b64_file(REPO / "tools" / "_runtime" / "tool_runtime.py")}
    tool_py = tool_dir / "tool.py"
    if tool_py.exists():
        files["tool.py"] = b64_file(tool_py)
    reqs = tool_dir / "requirements.txt"
    if reqs.exists():
        files["requirements.txt"] = b64_file(reqs)

    # Prefer the tool's own Dockerfile (installs its real dependencies); fall
    # back to the generic stdlib-only Dockerfile for any tool without one.
    dockerfile_path = tool_dir / "Dockerfile"
    dockerfile = dockerfile_path.read_bytes() if dockerfile_path.exists() else TOOL_DOCKERFILE.encode()

    c.deploy(program_id, tid, "docker", b64_bytes(dockerfile), files_b64=files)
    if not wait_for_image(program_id, tid, timeout=BUILD_TIMEOUT.get(tid, 300)):
        raise RuntimeError(f"image for tool {tid} not built in time")
    rec = dict(tool); rec.update({"machine_id": machine_id, "program_id": program_id,
                                  "entity_id": tid, "name": f"caspar__{tid}"})
    ok(f"tool creature deployed: {tid}  program={program_id}")
    return rec


def signal_tool(c: CasparSignalingClient, rec: dict) -> bool:
    tid = rec["tool_id"]
    # A broad payload so each tool can find the field it needs; tools ignore
    # the rest. Connectors with no creds still emit a (failing) TOOL_RESPONSE,
    # which is what the smoke test asserts on.
    signal = {"tool_id": tid, "function": rec.get("function", "invoke"),
              "payload": {"task": f"smoke-test {tid}", "query": "davinci caspar",
                          "url": "https://example.com", "ignore_https_errors": True,
                          "code": "result = 6*7", "sql": "SELECT 1 AS one",
                          "documents": ["davinci orchestrates caspar creatures"]}}
    vm_id = c.run_entity(rec["program_id"], rec["entity_id"],
                         params={"task.json": json.dumps(signal)}, ram_mb=256, max_exec_seconds=60)
    found, logs = c.wait_for_vm_log(vm_id, "TOOL_RESPONSE", timeout=90)
    tail = [_log_text(l) for l in logs if "TOOL_RESPONSE" in _log_text(l)]
    if found and tail:
        ok(f"signalled {tid} → {tail[-1][:120]}")
        return True
    bad(f"no response from {tid}; logs tail: {[_log_text(l) for l in logs[-4:]]}")
    return False


def deploy_davinci(c: CasparSignalingClient) -> dict:
    machine_id = c.create_machine_creature("m-davinci-agent")
    program_id = c.create_program(machine_id, "/davinci", "docker", "davinci agent")
    files = {"bundle.tar": b64_bytes(davinci_bundle_tar())}
    c.deploy(program_id, "davinci", "docker", b64_bytes(DAVINCI_DOCKERFILE.encode()), files_b64=files)
    if not wait_for_image(program_id, "davinci", timeout=360):
        raise RuntimeError("davinci image not built in time")
    ok(f"davinci creature deployed: program={program_id}")
    return {"machine_id": machine_id, "program_id": program_id, "entity_id": "davinci"}


def signal_davinci(c: CasparSignalingClient, davinci: dict, tool_recs: list) -> bool:
    task = {"objective": "Research the topic, retrieve context, and run a calculation, "
                         "using the available tool creatures.",
            "required_categories": [t["category"] for t in tool_recs]}
    config = {"node_host": NODE_HOST_FROM_VM, "node_port": NODE_PORT,
              "username": ADMIN_USER,
              "tools": [{"name": t["name"], "tool_id": t["tool_id"], "category": t["category"],
                         "risk": t["risk"], "description": t["description"],
                         "program_id": t["program_id"], "entity_id": t["entity_id"],
                         "function": t.get("function", "invoke"),
                         "requires_network": t.get("requires_network", False)}
                        for t in tool_recs]}
    vm_id = c.run_entity(davinci["program_id"], "davinci",
                         params={"task.json": json.dumps(task), "config.json": json.dumps(config)},
                         ram_mb=512, max_exec_seconds=240)
    info(f"davinci creature running (vm={vm_id}); waiting for DAVINCI_RESULT…")
    found, logs = c.wait_for_vm_log(vm_id, "DAVINCI_RESULT", timeout=220, poll=4)
    texts = [_log_text(l) for l in logs]
    result_line = next((t for t in texts if "DAVINCI_RESULT" in t), "")
    boot_line = next((t for t in texts if "DAVINCI_BOOT" in t), "")
    if boot_line:
        info("davinci boot: " + boot_line.split("DAVINCI_BOOT", 1)[1][:160])
    # Count how many tool creatures Davinci signalled (it triggers their VMs).
    signalled = sum(1 for t in texts if "tool ok" in t or "dispatched" in t)
    if found and result_line:
        try:
            result = json.loads(result_line.split("DAVINCI_RESULT", 1)[1].strip())
        except Exception:
            result = {}
        tool_results = [e for e in texts if "TOOL_RESPONSE" in e]
        ok(f"davinci finished: success={result.get('success')} "
           f"steps_done={result.get('plan', {}).get('progress', {}).get('done')} "
           f"tool_calls={result.get('budget', {}).get('tool_calls_used')}")
        ok(f"davinci-driven tool responses observed: {len(tool_results)}")
        return bool(result.get("budget", {}).get("tool_calls_used", 0))
    bad("davinci did not produce DAVINCI_RESULT")
    for t in texts[-8:]:
        print("   ", t[:160])
    return False


def main() -> int:
    info(f"connecting to Caspar node {NODE_HOST}:{NODE_PORT}")
    c = CasparSignalingClient(NODE_HOST, NODE_PORT, timeout=90).connect()
    c.login(ADMIN_USER)
    ok(f"logged in as {ADMIN_USER} (user_id={c.user_id})")

    results = {"tools_deployed": 0, "tools_signalled": 0, "davinci_deployed": False,
               "davinci_interacts": False}

    info("── Phase 1: deploy tool creatures ──")
    tool_recs = []
    for tool in TOOLS:
        rec = deploy_tool(c, tool)
        tool_recs.append(rec)
        results["tools_deployed"] += 1

    info("── Phase 2: signal tool creatures directly ──")
    for rec in tool_recs:
        if signal_tool(c, rec):
            results["tools_signalled"] += 1

    info("── Phase 3: deploy davinci creature ──")
    davinci = deploy_davinci(c)
    results["davinci_deployed"] = True

    info("── Phase 4: davinci signals tool creatures via caspar ──")
    results["davinci_interacts"] = signal_davinci(c, davinci, tool_recs)

    c.close()
    print("\n" + "=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60)
    success = (results["tools_deployed"] == len(TOOLS)
               and results["tools_signalled"] == len(TOOLS)
               and results["davinci_deployed"] and results["davinci_interacts"])
    (ok if success else bad)("OVERALL: " + ("PASS" if success else "FAIL"))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
