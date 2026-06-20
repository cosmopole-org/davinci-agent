"""Docker-creature entrypoint for Davinci on a Caspar node.

When Davinci is deployed as a ``docker`` creature and run via
``/programs/runEntity``, the Caspar node:

  * uploads any ``params`` to ``/app/input`` inside the container, and
  * captures the container's stdout as VM logs (readable with
    ``/machines/readVmLogs``).

This entrypoint reads its task from ``/app/input`` (or the ``DAVINCI_TASK`` env
var), runs the full Davinci agent loop with a streaming tracer, and prints
clearly-delimited, greppable JSON so a supervising client can assert on the
outcome.

A docker creature reaches the node and the outside world **only** through the
docker-host bridge gateway (``caspar_bridge``): it never opens the external
Caspar client protocol — that is reserved for host-side deployers/CLIs. When no
gateway is present (offline / local self-test) it runs with a built-in
capability set and an echo executor.

It produces three sentinel lines the deploy/test harness keys on:
    DAVINCI_BOOT   {...}      — environment + capability snapshot
    DAVINCI_TRACE  {...}      — one per trajectory event (streamed)
    DAVINCI_RESULT {...}      — the final run result
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from .attachments import Attachment, materialize_attachments
from .engine import DavinciAgent, EchoExecutor, ToolResult
from .mcp import ToolDescriptor, ToolRegistry
from .observability import Budget, Tracer
from .permissions import PermissionEngine, PermissionMode, Risk, ToolAction


INPUT_DIR = os.environ.get("DAVINCI_INPUT_DIR", "/app/input")


# --------------------------------------------------------------------------- #
# Session sandbox lifecycle
#
# When a `sandbox_pc` tool is part of the configured catalog, the per-session
# sandbox VM must be kept ALIVE while davinci is actively servicing this session
# (planning, calling tools, talking to the LLM, answering the user) and
# SUSPENDED (turned off, disk kept) once the response is delivered so the next
# prompt can wake it on the same persistent disk.
#
# The VM is driven through the SAME shared SandboxManager the tool uses, over
# this runtime's own bridge — no intermediary creature. We do this at the
# runtime boundary (around agent.run) rather than inside the engine: the engine
# is portable across executors, but a session VM's lifetime is a property of the
# deployed Caspar runtime.
# --------------------------------------------------------------------------- #


def _sandbox_enabled(tools_by_name: Dict[str, Dict[str, Any]]) -> bool:
    """True when a `sandbox_pc` tool is part of the configured catalog."""
    return any(spec.get("tool_id") == "sandbox_pc" for spec in tools_by_name.values())


class _SessionSandbox:
    """Keep the session's sandbox VM warm for the duration of one agent run.

    On entry: `start` the VM on its persistent disk before the first reasoner
    step. On exit: `suspend` it (idle / keep disk). All failures are swallowed —
    a sandbox that cannot be reached must never block the agent loop from
    running with the rest of its tools.
    """

    def __init__(self, bridge: Any, enabled: bool, session_id: str) -> None:
        self.session_id = session_id
        self.manager = None
        if bridge is not None and enabled:
            try:
                from .sandbox_core import SandboxManager
                self.manager = SandboxManager(bridge)
            except Exception as exc:  # noqa: BLE001
                print("DAVINCI_SANDBOX " + json.dumps({"init_error": repr(exc)[:160]}), flush=True)

    def _drive(self, action: str) -> None:
        if self.manager is None:
            return
        try:
            fn = self.manager.start if action == "start" else self.manager.suspend
            res = fn(self.session_id)
            print("DAVINCI_SANDBOX " + json.dumps(
                {"action": action, "session": self.session_id,
                 "ok": res.get("ok"), "vmId": res.get("vmId"), "runtime": res.get("runtime")}),
                flush=True)
        except Exception as exc:  # noqa: BLE001
            print("DAVINCI_SANDBOX " + json.dumps({"error": repr(exc)[:160], "action": action}),
                  flush=True)

    def __enter__(self) -> "_SessionSandbox":
        self._drive("start")
        return self

    def __exit__(self, *_exc: Any) -> None:
        # Idle path: turn off but KEEP the persistent disk. The next prompt for
        # this session will wake it transparently via the host's wake-on-exec.
        self._drive("suspend")


class BridgeCreatureExecutor:
    """Live executor that drives sibling tool creatures purely over the
    docker-host bridge gateway — Davinci's only channel to the outside world.

    To invoke a tool, Davinci pushes a ``creatures/signal`` to the tool's machine
    (via the node ``signalUser`` host function); the node delivers it to the
    tool's container over *its* gateway connection. The tool runs and signals its
    result back on the same key, which the node pushes onto Davinci's connection.
    Requests and replies are paired by ``correlationId``.
    """

    def __init__(self, bridge: Any, tools_by_name: Dict[str, Dict[str, Any]], my_id: str) -> None:
        import threading
        self.bridge = bridge
        self.tools = tools_by_name
        self.my_id = my_id
        self._lock = threading.Lock()
        self._waiters: Dict[str, threading.Event] = {}
        self._results: Dict[str, Any] = {}
        bridge.on_signal(self.handle_signal)

    def handle_signal(self, key: str, data: Any) -> None:
        if key != "creatures/signal" or not isinstance(data, dict):
            return
        if data.get("kind") != "tools/result":
            return
        cid = str(data.get("correlationId") or "")
        with self._lock:
            ev = self._waiters.get(cid)
            if ev is not None:
                self._results[cid] = data.get("result")
                ev.set()

    def execute(self, tool: ToolDescriptor, action: ToolAction) -> ToolResult:
        import threading
        import uuid
        spec = self.tools.get(tool.name)
        if not spec:
            return ToolResult(ok=False, output=None, error=f"unknown tool creature {tool.name}")
        target = spec.get("program_id") or spec.get("machine_id") or ""
        if not target:
            return ToolResult(ok=False, output=None, error=f"no target machine for tool {tool.name}")
        payload = {k: v for k, v in (action.args or {}).items() if v is not None}
        correlation_id = uuid.uuid4().hex
        # ``entityId`` is REQUIRED for the node to cold-spawn the right tool
        # creature: when no live tool connection exists the node's machine
        # listener resolves the per-entity docker image via
        # ``vmEntityPath::<machine>::<entityId>``. Without it the spawn falls back
        # to the program default and the tool image is never located, so the
        # invoke silently never runs and the caller times out. The node parses it
        # off the signal envelope (``stores::Send.entityId``).
        entity_id = spec.get("entity_id") or spec.get("tool_id", tool.name)
        packet = {
            "kind": "invoke",
            "entityId": entity_id,
            "correlationId": correlation_id,
            "reply_to": self.my_id,
            "tool_id": spec.get("tool_id", tool.name),
            "function": spec.get("function", "invoke"),
            "payload": payload,
        }
        ev = threading.Event()
        with self._lock:
            self._waiters[correlation_id] = ev
        try:
            ack = self.bridge.signal_user("creatures/signal", str(target), packet)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, output=None, error=f"bridge signal failed: {exc}")
        if isinstance(ack, dict) and ack.get("ok") is False:
            return ToolResult(ok=False, output={"ack": ack}, error="node rejected tool signal")
        # A sibling tool creature is usually *cold-spawned* on demand: the node
        # builds/starts its container, uploads the task, the tool connects to the
        # gateway, runs, and signals its result back. That whole round-trip can
        # take well over a minute under gVisor, so the default reply timeout is
        # generous (overridable per-tool via ``max_exec_seconds``).
        timeout = float(spec.get("max_exec_seconds", 240))
        delivered = ev.wait(timeout)
        with self._lock:
            self._waiters.pop(correlation_id, None)
            result = self._results.pop(correlation_id, None)
        if not delivered:
            return ToolResult(ok=False, output=None, error="tool creature result timed out")
        return ToolResult(ok=True, output={"tool": tool.name, "response": result})


def _read_task() -> Dict[str, Any]:
    """Resolve the task from /app/input files or environment."""
    # 1) explicit env task
    env_task = os.environ.get("DAVINCI_TASK")
    if env_task:
        return {"objective": env_task, "source": "env:DAVINCI_TASK"}

    # 2) a task.json dropped into the input dir
    task_json = os.path.join(INPUT_DIR, "task.json")
    if os.path.isfile(task_json):
        try:
            with open(task_json, encoding="utf-8") as fh:
                data = json.load(fh)
            data.setdefault("source", task_json)
            return data
        except (OSError, json.JSONDecodeError):
            pass

    # 3) any *.txt input treated as a free-form objective
    for txt in sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt"))):
        try:
            with open(txt, encoding="utf-8") as fh:
                return {"objective": fh.read().strip(), "source": txt}
        except OSError:
            continue

    # 4) default self-test objective
    return {"objective": "Run a self-test and report Davinci capabilities.",
            "source": "default"}


def _registry_from_tool_catalog(tools: List[Dict[str, Any]]) -> ToolRegistry:
    """Build a registry from a DAVINCI_TOOLS_JSON catalog of deployed tool creatures."""
    reg = ToolRegistry()
    for t in tools:
        reg.register(ToolDescriptor(
            name=t.get("name") or f"caspar__{t['tool_id']}",
            description=t.get("description", t["tool_id"]),
            category=t.get("category", "general"),
            risk=t.get("risk", "low"),
            requires_network=bool(t.get("requires_network", False)),
            server="caspar",
        ))
    return reg


def _read_config() -> Dict[str, Any]:
    """Read live-signalling config dropped into the input dir by the launcher."""
    path = os.path.join(INPUT_DIR, "config.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _build_registry() -> ToolRegistry:
    """Build the tool registry from the Caspar point catalog, or a demo set."""
    catalog_raw = os.environ.get("CASPAR_POINT_APPS_JSON", "").strip()
    if catalog_raw:
        try:
            from caspar_orchestrator import bootstrap_default_registry
            from .mcp import ToolRegistry as _TR
            caspar_reg = bootstrap_default_registry(point_catalog=json.loads(catalog_raw))
            if caspar_reg.list_tools():
                return _TR.from_caspar_registry(caspar_reg)
        except Exception:
            pass

    # Built-in demo capabilities so the creature is useful with zero config.
    reg = ToolRegistry()
    reg.register(ToolDescriptor("local__web_search", "Search the web for current information",
                                category="web_research", risk="low", requires_network=True))
    reg.register(ToolDescriptor("local__python_exec", "Execute Python in a sandbox",
                                category="execution", risk="medium"))
    reg.register(ToolDescriptor("local__git_tool", "Run git operations",
                                category="version_control", risk="medium"))
    reg.register(ToolDescriptor("local__vector_search", "Retrieve documents from a vector store",
                                category="knowledge_retrieval", risk="low"))
    return reg


def _capability_snapshot(registry: ToolRegistry) -> Dict[str, Any]:
    return {
        "version": _version(),
        "python": sys.version.split()[0],
        "tool_count": len(registry.all()),
        "categories": registry.categories(),
        "features": [
            "plan_and_execute", "react", "reflection", "replan_on_failure",
            "stuck_detection", "risk_gated_permissions", "guardrail_layering",
            "bounded_execution", "token_cost_budgeting", "event_sourced_tracing",
            "secret_masking", "hierarchical_instruction_memory", "mcp_tool_registry",
        ],
    }


def _version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "0.0.0"


def _run_agent(bridge: Any, task: Dict[str, Any], config: Dict[str, Any],
               attachments: List[Attachment]) -> tuple:
    """Build the agent from the (signal-delivered) task + config and run it.

    Returns ``(result, result_dict)``. The tool catalog and LLM config travel
    *with the task* over the signaling API; nothing is read from a task file in
    the gateway path.
    """
    objective = task.get("objective") or "Run a self-test and report Davinci capabilities."
    required = task.get("required_categories") or []
    tools = config.get("tools") or []

    executor: Any = EchoExecutor()
    discovery = "none"
    live = False
    tools_by_name: Dict[str, Dict[str, Any]] = {}

    if bridge is not None:
        # Use the node-assigned identity reported in the WELCOME handshake — the
        # container never declares its own id. Davinci drives sibling tool
        # creatures purely by signalling their live VMs over the gateway.
        my_id = bridge.machine_id or bridge.program_id or ""
        tools_by_name = {t.get("name") or f"caspar__{t['tool_id']}": t for t in tools}
        registry = _registry_from_tool_catalog(tools) if tools else _build_registry()
        executor = BridgeCreatureExecutor(bridge, tools_by_name, my_id)
        discovery = "bridge"
        live = True
        if not required:
            required = list(dict.fromkeys(
                t.get("category", "general") for t in tools_by_name.values()))[:4]
    else:
        # Offline / local self-test: built-in capability set, no signalling.
        registry = _build_registry()

    # LLM backbone: use Gemini as the reasoner when an API key is supplied via
    # the task's config. Any failure leaves the deterministic HeuristicReasoner.
    reasoner: Any = None
    llm_provider = "heuristic"
    try:
        from .gemini_reasoner import reasoner_from_config
        reasoner = reasoner_from_config(config)
        if reasoner is not None:
            llm_provider = "gemini:" + ",".join(reasoner.models)
    except Exception as exc:  # noqa: BLE001 — never block boot on LLM wiring
        print("DAVINCI_BOOT " + json.dumps({"llm_init_error": repr(exc)[:160]}), flush=True)

    snapshot = _capability_snapshot(registry)
    snapshot["live_signaling"] = live
    snapshot["discovery"] = discovery
    snapshot["llm_provider"] = llm_provider
    task_summary = {k: v for k, v in task.items() if k not in ("attachments", "config")}
    if task.get("attachments"):
        task_summary["attachments"] = [
            {kk: vv for kk, vv in (a.items() if isinstance(a, dict) else [])
             if kk in ("name", "mime_type", "description", "path")}
            for a in task["attachments"]
        ]
    print("DAVINCI_BOOT " + json.dumps({"task": task_summary, "capabilities": snapshot}), flush=True)

    tracer = Tracer(stream=True)  # emits DAVINCI_TRACE lines as it goes
    mode = PermissionMode(os.environ.get("DAVINCI_PERMISSION_MODE", "auto"))
    # Risk ceiling for autonomous (auto-mode) tool execution. The agent runs
    # unattended as a creature, so network/automation tools (web fetch, headless
    # browser) are legitimately part of the job and must be runnable without a
    # human reviewer. Configurable (task config ``risk_ceiling`` or the
    # DAVINCI_RISK_CEILING env); defaults to ``high`` so a deployed agent can
    # actually use its high-risk tool creatures; lower it to harden a deployment.
    _ceiling_name = str(
        config.get("risk_ceiling")
        or os.environ.get("DAVINCI_RISK_CEILING", "high")
    ).lower()
    risk_ceiling = {"low": Risk.LOW, "medium": Risk.MEDIUM, "high": Risk.HIGH}.get(
        _ceiling_name, Risk.HIGH)
    # LLM-backed planning: when the reasoner can decompose the objective itself
    # (Gemini), plan with concrete, executable steps instead of generic
    # category placeholders, so each step yields real tool arguments.
    planner = None
    if reasoner is not None and hasattr(reasoner, "make_plan"):
        from .planning import Planner
        planner = Planner(strategy=lambda obj, cats: reasoner.make_plan(obj, cats, registry))

    agent = DavinciAgent(
        registry=registry,
        permissions=PermissionEngine(mode=mode, risk_ceiling=risk_ceiling),
        reasoner=reasoner,  # None -> engine default (HeuristicReasoner)
        executor=executor,
        planner=planner,    # None -> engine default (deterministic Planner)
        tracer=tracer,
        budget=Budget(max_steps=int(os.environ.get("DAVINCI_MAX_STEPS", "20"))),
    )

    sandbox_enabled = bridge is not None and _sandbox_enabled(tools_by_name)
    session_id = (
        task.get("session_id")
        or task.get("sessionId")
        or (f"davinci-{bridge.session_id}" if bridge and getattr(bridge, "session_id", None) else "davinci-default")
    )
    with _SessionSandbox(bridge, sandbox_enabled, session_id):
        result = agent.run(objective, required_categories=required,
                           attachments=attachments)
    result_dict = result.to_dict()
    print("DAVINCI_RESULT " + json.dumps(result_dict), flush=True)
    return result, result_dict


def _wait_for_task_signal(bridge: Any, timeout: float) -> tuple:
    """Block until the agent's task arrives over the signaling API.

    The task is delivered as a ``/creatures/signal`` (pvp) which the node wraps
    as ``StoresSend{user, data:"<json>"}`` and pushes onto our gateway
    connection. We capture it on the bridge reader thread (no host calls there)
    and hand it to the main thread, which runs the agent loop.

    Returns ``(task, reply_to, correlation_id)`` or ``(None, None, None)``.
    """
    import threading

    box: Dict[str, Any] = {}
    ev = threading.Event()

    def on_signal(key: str, data: Any) -> None:
        if key != "creatures/signal" or not isinstance(data, dict):
            return
        inner = data.get("data")
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:  # noqa: BLE001
                return
        if not isinstance(inner, dict):
            inner = data
        # Only act on a task delivery, not on unrelated signals.
        if inner.get("kind") != "task" and "objective" not in inner:
            return
        box["task"] = inner
        box["reply_to"] = inner.get("reply_to") or (data.get("user") or {}).get("id")
        box["correlationId"] = inner.get("correlationId")
        ev.set()

    bridge.on_signal(on_signal)
    print("DAVINCI_READY " + json.dumps(
        {"machine_id": getattr(bridge, "machine_id", ""),
         "program_id": getattr(bridge, "program_id", ""),
         "session": bridge.session_id, "ts": time.time()}), flush=True)
    if not ev.wait(timeout):
        return None, None, None
    return box.get("task"), box.get("reply_to"), box.get("correlationId")


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    # Gateway path: when the node injected CASPAR_GATEWAY_* env, this container is
    # a gateway-managed docker creature and the bridge is its *only* channel. The
    # agent's task (objective + tool catalog + LLM config) is delivered over the
    # signaling API — never via an uploaded task file.
    bridge = None
    try:
        from .caspar_bridge import bridge_from_env
        bridge = bridge_from_env()
    except Exception as exc:  # noqa: BLE001 — fall back to the offline path
        print("DAVINCI_BOOT " + json.dumps({"bridge_init_error": repr(exc)[:160]}), flush=True)
        bridge = None

    if bridge is not None:
        print("DAVINCI_BRIDGE " + json.dumps({
            "connected": True, "session": bridge.session_id,
            "vm_id": os.environ.get("CASPAR_VM_ID", ""),
            "machine_id": getattr(bridge, "machine_id", "")}), flush=True)
        task, reply_to, correlation_id = _wait_for_task_signal(
            bridge, float(os.environ.get("DAVINCI_TASK_WAIT", "600")))
        if not task:
            print("DAVINCI_RESULT " + json.dumps(
                {"success": False, "error": "no task signal received within wait window"}), flush=True)
            try:
                bridge.close()
            except Exception:  # noqa: BLE001
                pass
            return 2
        config = task.get("config") or {}
        attachments = materialize_attachments(task, INPUT_DIR)
        if attachments:
            print("DAVINCI_ATTACHMENTS " + json.dumps(
                {"count": len(attachments), "items": [a.to_dict() for a in attachments]}), flush=True)
        result, result_dict = _run_agent(bridge, task, config, attachments)
        # Signal the final result back to the requester over the same API.
        if reply_to:
            try:
                bridge.signal_user("creatures/signal", str(reply_to),
                                   {"kind": "davinci/result", "correlationId": correlation_id,
                                    "result": result_dict})
            except Exception as exc:  # noqa: BLE001
                print("DAVINCI_BOOT " + json.dumps({"reply_error": repr(exc)[:160]}), flush=True)
        try:
            bridge.close()
        except Exception:  # noqa: BLE001
            pass
        return 0 if result.success else 2

    # Offline / local self-test (no gateway): read the task + config from files.
    task = _read_task()
    config = _read_config()
    attachments = materialize_attachments(task, INPUT_DIR)
    if attachments:
        print("DAVINCI_ATTACHMENTS " + json.dumps(
            {"count": len(attachments), "items": [a.to_dict() for a in attachments]}), flush=True)
    result, _ = _run_agent(None, task, config, attachments)
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
