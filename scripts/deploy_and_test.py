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
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from davinci.caspar_signaling import CasparSignalingClient, _log_text  # noqa: E402

# A short per-run tag keeps machine-creature usernames unique so the harness can
# be re-run against a node that already persisted creatures from a prior run
# (the node rejects duplicate creature usernames). Override with CASPAR_RUN_TAG.
RUN_TAG = os.environ.get("CASPAR_RUN_TAG") or uuid.uuid4().hex[:8]

NODE_HOST = os.environ.get("CASPAR_NODE_HOST", "127.0.0.1")
NODE_PORT = int(os.environ.get("CASPAR_NODE_PORT", "8074"))
# Address the *creature* uses to reach the node from inside the docker network.
# Defaults to the ``kasper`` bridge gateway the node attaches creatures to.
NODE_HOST_FROM_VM = os.environ.get("CASPAR_NODE_HOST_FROM_VM", "172.18.0.1")
ADMIN_USER = "davinci_admin"

# LLM backbone — the Gemini API key is taken from the environment only, never
# hard-coded or committed. When set, it is passed to the Davinci creature via
# its config.json so the agent reasons with Gemini.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODELS = [m.strip() for m in os.environ.get("GEMINI_MODELS", "").split(",") if m.strip()]

# Provider-neutral LLM selection: the agent's reasoning is cross-LLM, so the
# harness lets you pick the backbone via env without code changes. LLM_PROVIDER
# (gemini|anthropic|openai|grok|agentrouter) + the matching <PROVIDER>_API_KEY / _MODELS.
# Defaults to Gemini so existing runs are unchanged.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").strip().lower()
_PROVIDER_ENVS = {
    "gemini":      ("GEMINI_API_KEY",      "GEMINI_MODELS"),
    "anthropic":   ("ANTHROPIC_API_KEY",   "ANTHROPIC_MODELS"),
    "openai":      ("OPENAI_API_KEY",      "OPENAI_MODELS"),
    "grok":        ("GROK_API_KEY",        "GROK_MODELS"),
    "agentrouter": ("AGENTROUTER_API_KEY", "AGENTROUTER_MODELS"),
    "openrouter":  ("OPENROUTER_API_KEY",  "OPENROUTER_MODELS"),
}


def llm_config() -> dict:
    """Build the LLM portion of the davinci creature config from the environment.

    Always carries any ``GEMINI_*`` values (backward compatible), and — when
    ``LLM_PROVIDER`` names a different provider with a key set — adds the generic
    ``llm_provider`` + ``<provider>_api_key`` / ``<provider>_models`` keys the
    creature's ``reasoner_from_config`` understands. Keys are read from the
    environment of *this* harness only; never written to disk or committed.
    """
    cfg: dict = {}
    if GEMINI_API_KEY:
        cfg["gemini_api_key"] = GEMINI_API_KEY
        cfg["gemini_models"] = GEMINI_MODELS or None
    if LLM_PROVIDER in _PROVIDER_ENVS:
        key_env, models_env = _PROVIDER_ENVS[LLM_PROVIDER]
        key = os.environ.get(key_env, "").strip()
        if key:
            cfg["llm_provider"] = LLM_PROVIDER
            cfg[f"{LLM_PROVIDER}_api_key"] = key
            models = [m.strip() for m in os.environ.get(models_env, "").split(",") if m.strip()]
            if models:
                cfg[f"{LLM_PROVIDER}_models"] = models
    return cfg


def llm_bake_env() -> dict:
    """The LLM backbone as **environment variables** to bake into a creature image.

    Same selection as :func:`llm_config` but keyed by the provider env-var names
    (``GEMINI_API_KEY``, ``LLM_PROVIDER``, ``<PROVIDER>_API_KEY`` …) so the
    deployed davinci creature reasons with a real provider without the caller
    supplying an LLM config on each signal. Read from this harness's environment
    only; empty when no key is set.
    """
    env: dict = {}
    if GEMINI_API_KEY:
        env["GEMINI_API_KEY"] = GEMINI_API_KEY
        if GEMINI_MODELS:
            env["GEMINI_MODELS"] = ",".join(GEMINI_MODELS)
    if LLM_PROVIDER in _PROVIDER_ENVS:
        key_env, models_env = _PROVIDER_ENVS[LLM_PROVIDER]
        key = os.environ.get(key_env, "").strip()
        if key:
            env["LLM_PROVIDER"] = LLM_PROVIDER
            env[key_env] = key
            # Bake the model. The runtime resolves it from either the plural
            # <PROVIDER>_MODELS or the singular <PROVIDER>_MODEL env var, so carry
            # whichever the operator set — and derive the singular from the plural
            # (reasoner_from_config reads the singular from the environment).
            model_env = f"{LLM_PROVIDER.upper()}_MODEL"
            models = os.environ.get(models_env, "").strip()
            single = os.environ.get(model_env, "").strip()
            if models:
                env[models_env] = models
            if single:
                env[model_env] = single
            elif models:
                env[model_env] = models.split(",")[0].strip()
    return env

# The per-space sandbox creature talks to the Vercel Sandbox REST API. Its token
# is baked into the tool image (never sent in a signal payload), read from this
# harness's environment only — never written to the repo or committed.
# Every name tools/vercel_sandbox/tool.py reads. All three token spellings are
# here on purpose: the tool accepts any of them, so baking only VERCEL_TOKEN
# would let an operator who set VERCEL_API_TOKEN deploy a creature that looks
# healthy and then refuses every call for want of credentials.
SANDBOX_ENV_KEYS = (
    "VERCEL_TOKEN", "VERCEL_API_TOKEN", "VERCEL_ACCESS_TOKEN",
    "VERCEL_TEAM_ID", "VERCEL_PROJECT_ID", "VERCEL_API_BASE",
    "VERCEL_SANDBOX_RUNTIME", "VERCEL_SANDBOX_TIMEOUT_MS", "VERCEL_SANDBOX_VCPUS",
    "VERCEL_SANDBOX_PREFIX", "VERCEL_SANDBOX_MAX_OUTPUT", "VERCEL_SANDBOX_MAX_READ_BYTES",
    "VERCEL_SANDBOX_EXEC_TIMEOUT_MS", "VERCEL_SANDBOX_HTTP_TIMEOUT",
    "VERCEL_SANDBOX_SESSION_TTL",
)


def sandbox_bake_env() -> dict:
    """Vercel credentials + tuning to bake into the vercel_sandbox tool image."""
    return {k: os.environ[k].strip() for k in SANDBOX_ENV_KEYS
            if os.environ.get(k, "").strip()}


# In hardened environments outbound HTTPS is intercepted by an egress gateway
# whose CA must be trusted inside the creature containers (otherwise Gemini /
# web tools fail TLS verification). We ship the host CA bundle into each image.
CA_BUNDLE_PATH = os.environ.get("CASPAR_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")

TOOLS_DIR = REPO / "tools"


def _ca_bundle_bytes():
    """Read the host CA bundle (incl. egress-gateway CAs), or ``None``."""
    try:
        data = Path(CA_BUNDLE_PATH).read_bytes()
        return data if data.strip() else None
    except OSError:
        return None


# Appended to every creature Dockerfile so the egress-gateway CA is trusted and
# the standard TLS env vars point at the baked-in bundle (Python, requests, Node).
CA_DOCKERFILE_SNIPPET = (
    "COPY ca-certificates.crt /app/ca-certificates.crt\n"
    "ENV SSL_CERT_FILE=/app/ca-certificates.crt "
    "REQUESTS_CA_BUNDLE=/app/ca-certificates.crt "
    "NODE_EXTRA_CA_CERTS=/app/ca-certificates.crt\n"
)


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


def tool_full_metadata(tool_id: str) -> dict:
    """The tool's complete point.metadata.json (MCP manifest incl. tools[].args)."""
    path = TOOLS_DIR / tool_id / "point.metadata.json"
    try:
        return json.loads(path.read_text())
    except OSError:
        return {}


_JSON_TYPES = {"STRING": "string", "INT": "integer", "INTEGER": "integer",
               "NUMBER": "number", "FLOAT": "number", "BOOL": "boolean",
               "BOOLEAN": "boolean", "OBJECT": "object", "ARRAY": "array", "LIST": "array"}


def tool_arg_schema(tool_id: str) -> dict:
    """JSON-Schema ``properties`` for a tool's primary function, derived from its
    point.metadata.json ``tools[].args``.

    Carried in the tool catalog so Davinci's LLM reasoner emits each tool's real
    argument names (e.g. python_exec's ``code``, web_search's ``query``) instead
    of falling back to a generic ``task`` — without the schema the model has no
    way to know the contract and the fallback ships the objective text as code.
    """
    funcs = tool_full_metadata(tool_id).get("tools") or []
    args = (funcs[0].get("args") or {}) if funcs else {}
    props: dict = {}
    for name, spec in args.items():
        spec = spec or {}
        props[name] = {"type": _JSON_TYPES.get(str(spec.get("type", "STRING")).upper(), "string"),
                       "description": spec.get("desc", "")}
    return props


def deploy_wasm_creature(c: CasparSignalingClient, namespace: str, wasm_path: Path,
                         entity_id: str = "main") -> dict:
    """Deploy a WASM miniapp creature (e.g. the Decillion ``stores`` namespace).

    Mirrors decillionai-server/bench/deploy.py: create a machine creature, a
    ``wasm`` program under it, then deploy the .wasm as the program's entity.
    Returns ``{machine_id, program_id, entity_id}`` — the miniapp signalling
    target (creatureId = machine_id, programId = program_id).
    """
    machine_id = c.create_machine_creature(f"m-{namespace}-{RUN_TAG}")
    program_id = c.create_program(machine_id, f"/{namespace}", "wasm", f"{namespace} miniapp")
    c.deploy(program_id, entity_id, "wasm", b64_file(wasm_path),
             metadata={"namespace": namespace})
    ok(f"{namespace} miniapp deployed: creature={machine_id} program={program_id}")
    return {"machine_id": machine_id, "program_id": program_id, "entity_id": entity_id}


# Tool creatures to deploy (every tool under tools/ with metadata).
TOOLS = _discover_tools()

# Heavier images (system packages / browser binaries) need a longer build window.
BUILD_TIMEOUT = {"browser_automation": 900, "sql_query": 480, "vector_search": 480,
                 "calendar_connector": 480}

GREEN, RED, YELLOW, CYAN, NC = "\033[0;32m", "\033[0;31m", "\033[0;33m", "\033[0;36m", "\033[0m"


def info(m): print(f"{CYAN}[deploy]{NC} {m}", flush=True)
def ok(m):   print(f"{GREEN}[ ok ]{NC} {m}", flush=True)
def warn(m): print(f"{YELLOW}[warn]{NC} {m}", flush=True)
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


# Label stamped into every creature image, carrying a digest of the exact build
# context it was built from. It is what lets a redeploy tell "the node already
# built this" from "the node has not finished building yet" — see
# `wait_for_image`, where guessing between those two used to cost a full timeout.
CONTEXT_LABEL = "org.decillion.build-context"


def context_digest(dockerfile: bytes, files_b64: dict) -> str:
    """sha256 over the whole build context (Dockerfile + every shipped file)."""
    h = hashlib.sha256()
    h.update(dockerfile)
    for name in sorted(files_b64 or {}):
        h.update(name.encode())
        h.update(files_b64[name].encode())
    return h.hexdigest()


def stamp_context(dockerfile: bytes, files_b64: dict) -> tuple:
    """Append the context LABEL to a Dockerfile. Returns (dockerfile, digest).

    The label is derived from the context *without* it, so it is stable, and
    because it lands in the image config a changed context always produces a
    different image id — which is what makes waiting on the id meaningful.
    """
    digest = context_digest(dockerfile, files_b64)
    return dockerfile + f'\nLABEL {CONTEXT_LABEL}="{digest}"\n'.encode(), digest


def docker_image_context(program_id: str, entity_id: str) -> str:
    """The context digest baked into the current image, or "" when absent."""
    image = f"{program_id.replace('@', '_')}/{entity_id}"
    try:
        out = subprocess.run(
            ["docker", "inspect", "--format",
             '{{index .Config.Labels "' + CONTEXT_LABEL + '"}}', image],
            capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def docker_image_id(program_id: str, entity_id: str) -> str:
    """Current image ID for a program/entity tag, or "" when it doesn't exist.

    A successful docker build re-tags the image to a NEW id when the build
    context changed; a fully-cached (no-op) rebuild keeps the SAME id. We use
    this to tell a real rebuild from a stale tag on a redeploy.
    """
    image = f"{program_id.replace('@', '_')}/{entity_id}"
    try:
        out = subprocess.run(["docker", "images", "--no-trunc", "--format", "{{.ID}}", image],
                             capture_output=True, text=True, timeout=15)
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return lines[0] if lines else ""
    except Exception:
        return ""


def wait_for_image(program_id: str, entity_id: str, timeout: int = 240,
                   prev_image_id: Optional[str] = None,
                   expect_context: Optional[str] = None) -> bool:
    """Wait until the node has (re)built the entity's docker image.

    The node builds asynchronously and only re-tags on success, so on a REDEPLOY
    the old tag is present the whole time — a plain existence check returns
    instantly and runEntity would recreate the container from the PRE-rebuild
    image. So we wait for the image to actually change.

    ``expect_context`` is the digest of the context we just deployed (see
    :func:`stamp_context`) and is what makes that wait *terminate*. Without it
    the only signal is the image id, and a fully-cached rebuild — identical
    code, so docker re-tags the same id — is indistinguishable from a build
    still in progress: the loop then burned the entire timeout (6 minutes for
    davinci) on every no-op redeploy before shrugging and proceeding. Since the
    digest is a LABEL, a changed context always yields a different image, so
    "the image already carries this digest" means the build is genuinely done.
    """
    image = f"{program_id.replace('@', '_')}/{entity_id}"
    deadline = time.time() + timeout
    if prev_image_id or expect_context:
        info(f"waiting for node to REBUILD image {image} (≤{timeout}s)…")
        while time.time() < deadline:
            if expect_context and docker_image_context(program_id, entity_id) == expect_context:
                ok(f"image built from the deployed context: {image}")
                return True
            cur = docker_image_id(program_id, entity_id)
            if prev_image_id and cur and cur != prev_image_id:
                ok(f"image rebuilt: {image} -> {cur[:19]}")
                return True
            time.sleep(3)
        # Reaching here with a context digest means the node never produced an
        # image carrying it — a real build failure, not a cached no-op. It is
        # still only a warning because a host where `docker` is not queryable
        # (the deployer does not always share the node's docker socket) reports
        # empty for everything, and that must not fail an otherwise fine deploy.
        warn(f"image {image} did not change after {timeout}s — proceeding with the "
             f"current image; check the node's build logs if the entity misbehaves")
        return True
    info(f"waiting for node to build image {image} (≤{timeout}s)…")
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

def deploy_tool(c: CasparSignalingClient, tool: dict, *, program_id: Optional[str] = None,
                machine_id: str = "", bake_env: Optional[dict] = None) -> dict:
    """Deploy (or re-deploy) one tool as its own docker creature.

    Passing ``program_id`` redeploys a FRESH entity onto an existing program
    instead of minting a new creature — the same reuse path ``deploy_davinci``
    takes, so a tool that Nest already references by id (the per-space sandbox)
    keeps that id across redeploys and no space is left pointing at a dead
    program.
    """
    tid = tool["tool_id"]
    tool_dir = REPO / "tools" / tid
    prev_image_id = None
    if program_id:
        info(f"redeploying tool {tid} onto existing program {program_id} — no new creature")
        prev_image_id = docker_image_id(program_id, tid)
    else:
        machine_id = c.create_machine_creature(f"m-tool-{tid}-{RUN_TAG}")
        program_id = c.create_program(machine_id, f"/tools/{tid}", "docker", f"tool {tid}")

    # Build context: the shared runtime + the tool's real implementation, plus
    # its requirements.txt so the tool's own Dockerfile can install its deps.
    files = {"tool_runtime.py": b64_file(REPO / "tools" / "_runtime" / "tool_runtime.py")}
    # Ship the docker-host bridge client so the tool can reach the node's host
    # functions (HTTP/DB) and signal its result back over the gateway.
    bridge_py = REPO / "davinci" / "caspar_bridge.py"
    if bridge_py.exists():
        files["caspar_bridge.py"] = b64_file(bridge_py)
    # Ship the shared sandbox manager so the sandbox_pc tool can drive per-session
    # VMs directly through the vmm host APIs (no intermediary creature). Harmless
    # for other tools, which simply never import it.
    sandbox_core_py = REPO / "davinci" / "sandbox_core.py"
    if sandbox_core_py.exists():
        files["sandbox_core.py"] = b64_file(sandbox_core_py)
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

    # Trust the egress-gateway CA inside the image so network tools can make
    # outbound HTTPS calls through the intercepting proxy.
    ca = _ca_bundle_bytes()
    if ca is not None:
        files["ca-certificates.crt"] = b64_bytes(ca)
        dockerfile = dockerfile + b"\n" + CA_DOCKERFILE_SNIPPET.encode()

    # browser_automation: optionally bake in an authenticated-session cookie jar
    # (BROWSER_COOKIES_FILE) and an outbound proxy (BROWSER_PROXY) so the headless
    # browser can reach + authenticate to sites that are geo/route-blocked or
    # auth-gated on the creature's direct egress. Both are read from this harness's
    # environment only — never written to the repo or committed.
    if tid == "browser_automation":
        cookies_file = os.environ.get("BROWSER_COOKIES_FILE", "").strip()
        if cookies_file and Path(cookies_file).exists():
            files["cookies.json"] = b64_file(Path(cookies_file))
        proxy = os.environ.get("BROWSER_PROXY", "").strip()
        if proxy:
            dockerfile = dockerfile + f"\nENV BROWSER_PROXY={proxy}\n".encode()

    # Code/HTTP tools fetch from sites that may be geo/route-blocked on the
    # creature's direct egress (e.g. Binance returns connection-refused from a
    # datacenter IP). Bake the proxy as HTTPS_PROXY/HTTP_PROXY so the `requests`
    # library inside python_exec (and the HTTP tools) routes through it. The
    # python_exec subprocess inherits the container env, so user code picks it up.
    tool_proxy = (os.environ.get("TOOL_HTTP_PROXY")
                  or os.environ.get("BROWSER_PROXY", "")).strip()
    if tool_proxy and tid in ("python_exec", "fetch_url", "web_search"):
        dockerfile = dockerfile + (
            f"\nENV HTTPS_PROXY={tool_proxy} HTTP_PROXY={tool_proxy} "
            f"https_proxy={tool_proxy} http_proxy={tool_proxy}\n").encode()

    # Per-tool credentials (e.g. the vercel_sandbox API token) are baked into the
    # image so they never travel in a signal payload an agent could influence.
    dockerfile = dockerfile + _bake_env_snippet(bake_env or {}).encode()
    dockerfile, digest = stamp_context(dockerfile, files)

    # An image already built from this exact context needs no rebuild and no
    # wait — the deploy below only (re)registers the entity with the node.
    if prev_image_id and docker_image_context(program_id, tid) == digest:
        c.deploy(program_id, tid, "docker", b64_bytes(dockerfile), files_b64=files)
        ok(f"tool creature {tid} already built from this context: program={program_id}")
    else:
        c.deploy(program_id, tid, "docker", b64_bytes(dockerfile), files_b64=files)
        if not wait_for_image(program_id, tid, timeout=BUILD_TIMEOUT.get(tid, 300),
                              prev_image_id=prev_image_id, expect_context=digest):
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


def _bake_env_snippet(env: dict) -> str:
    """A Dockerfile ``ENV`` line baking key=value pairs into the creature image.

    Used to embed the LLM backbone credentials (e.g. ``GEMINI_API_KEY``) into the
    davinci agent creature so it reasons with a real provider even when the
    *caller* (the Decillion backend proxy) signals it without an LLM config —
    ``client_from_config`` falls back to these env vars. The value never touches
    disk on the host and never leaves the node's local image store.
    """
    if not env:
        return ""
    parts = []
    for k, v in env.items():
        if v is None or str(v) == "":
            continue
        # Quote the value; escape backslashes and double-quotes for Dockerfile ENV.
        sv = str(v).replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'{k}="{sv}"')
    return ("ENV " + " ".join(parts) + "\n") if parts else ""


def deploy_davinci(c: CasparSignalingClient, bake_env: dict = None,
                   program_id: str = None, entity_id: str = "davinci") -> dict:
    """Deploy (or re-deploy) the davinci docker entity.

    A Caspar creature program is permanent and re-deployable: an updated entity
    — a new Dockerfile plus payload/files — can be pushed onto the SAME program
    id via ``deployEntity`` any number of times. So when ``program_id`` is given
    (the id already recorded in the manifest), we skip minting a new
    creature/program and just redeploy a FRESH entity onto it. This picks up
    davinci code changes while keeping the program id stable — every already
    deployed agent proxy still targets a valid davinci (no orphaning). When
    ``program_id`` is None, a new creature + program are created (first-time
    bootstrap).
    """
    prev_image_id = None
    if program_id:
        info(f"redeploying davinci entity onto existing program {program_id} "
             f"(entity {entity_id}) — no new creature")
        machine_id = ""
        # Capture the current image id BEFORE the rebuild so wait_for_image can
        # wait for it to actually change (the node rebuilds async and keeps the
        # old tag until the new build succeeds).
        prev_image_id = docker_image_id(program_id, entity_id)
    else:
        machine_id = c.create_machine_creature(f"m-davinci-agent-{RUN_TAG}")
        program_id = c.create_program(machine_id, "/davinci", "docker", "davinci agent")
        entity_id = "davinci"
    files = {"bundle.tar": b64_bytes(davinci_bundle_tar())}
    dockerfile = DAVINCI_DOCKERFILE
    ca = _ca_bundle_bytes()
    if ca is not None:
        files["ca-certificates.crt"] = b64_bytes(ca)
        dockerfile = dockerfile + CA_DOCKERFILE_SNIPPET
    dockerfile = dockerfile + _bake_env_snippet(bake_env or {})
    stamped, digest = stamp_context(dockerfile.encode(), files)
    already_current = bool(prev_image_id) and docker_image_context(program_id, entity_id) == digest
    c.deploy(program_id, entity_id, "docker", b64_bytes(stamped), files_b64=files)
    if already_current:
        # Same code as the running image: the node's rebuild is a cached no-op,
        # so there is nothing to wait for. This is the case that used to burn the
        # whole DAVINCI_REBUILD_TIMEOUT on every redeploy of unchanged davinci.
        ok(f"davinci image already built from this context — no rebuild to wait for")
    else:
        _rebuild_timeout = int(os.environ.get("DAVINCI_REBUILD_TIMEOUT", "360"))
        if not wait_for_image(program_id, entity_id, timeout=_rebuild_timeout,
                              prev_image_id=prev_image_id, expect_context=digest):
            raise RuntimeError("davinci image not built in time")
    ok(f"davinci creature deployed: program={program_id} entity={entity_id}")
    return {"machine_id": machine_id, "program_id": program_id, "entity_id": entity_id}


def signal_davinci(c: CasparSignalingClient, davinci: dict, tool_recs: list) -> bool:
    task = {"objective": "Research the topic, retrieve context, and run a calculation, "
                         "using the available tool creatures.",
            "required_categories": [t["category"] for t in tool_recs]}
    config = {"node_host": NODE_HOST_FROM_VM, "node_port": NODE_PORT,
              "username": ADMIN_USER,
              # LLM backbone for the agent's reasoning (provider-neutral). Keys
              # come from the environment of *this* harness only — never committed.
              **llm_config(),
              "tools": [{"name": t["name"], "tool_id": t["tool_id"], "category": t["category"],
                         "risk": t["risk"], "description": t["description"],
                         "program_id": t["program_id"], "entity_id": t["entity_id"],
                         "function": t.get("function", "invoke"),
                         "arg_schema": tool_arg_schema(t["tool_id"]),
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
        rec = deploy_tool(c, tool,
                          bake_env=sandbox_bake_env() if tool["tool_id"] == "vercel_sandbox" else None)
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
