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
from typing import Any, Dict, List, Optional

from .engine import DavinciAgent, EchoExecutor, ToolResult
from .mcp import ToolDescriptor, ToolRegistry
from .observability import Budget, Tracer
from .permissions import PermissionEngine, PermissionMode, Risk, ToolAction


INPUT_DIR = os.environ.get("DAVINCI_INPUT_DIR", "/app/input")


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
        packet = {
            "kind": "invoke",
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
        timeout = float(spec.get("max_exec_seconds", 75))
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


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    task = _read_task()
    objective = task.get("objective") or "Run a self-test and report Davinci capabilities."
    required = task.get("required_categories") or []

    # Live mode: signal sibling tool creatures through the node over the bridge.
    # Configuration (an optional tool catalog) is delivered as an input file by
    # the signal that started us.
    config = _read_config()
    tools = config.get("tools") or []

    executor: Any = EchoExecutor()
    discovery = "none"
    mode_default = "auto"
    live = False

    # Gateway path: when the node injected CASPAR_GATEWAY_* env, this container
    # is a gateway-managed docker creature and the bridge is its *only* route to
    # the outside world. A docker creature is bridge-or-nothing — it never opens
    # the external Caspar client protocol (that is reserved for host-side
    # deployers/CLIs, not for sandboxed creatures).
    bridge = None
    try:
        from .caspar_bridge import bridge_from_env
        bridge = bridge_from_env()
    except Exception as exc:  # noqa: BLE001 — fall back to the offline path
        print("DAVINCI_BOOT " + json.dumps({"bridge_init_error": repr(exc)[:160]}), flush=True)
        bridge = None

    if bridge is not None:
        # Use the node-assigned identity reported in the WELCOME handshake — the
        # container never declares its own id.
        my_id = bridge.machine_id or bridge.program_id or ""
        tools_by_name = {t.get("name") or f"caspar__{t['tool_id']}": t for t in tools}
        registry = _registry_from_tool_catalog(tools) if tools else _build_registry()
        executor = BridgeCreatureExecutor(bridge, tools_by_name, my_id)
        discovery = "bridge"
        live = True
        if not required:
            required = list(dict.fromkeys(
                t.get("category", "general") for t in tools_by_name.values()))[:4]
        print("DAVINCI_BRIDGE " + json.dumps({
            "connected": True, "session": bridge.session_id, "vm_id": os.environ.get("CASPAR_VM_ID", ""),
            "tools": list(tools_by_name.keys())}), flush=True)
    else:
        # Not gateway-managed (offline / local self-test): run with the built-in
        # capability set. No external Caspar client connection is ever opened
        # from inside a creature.
        registry = _build_registry()
        live = False

    # LLM backbone: use Gemini as the reasoner when an API key is supplied via
    # the live config (config.json) or the GEMINI_API_KEY env var. Any failure to
    # construct it leaves the deterministic HeuristicReasoner in place.
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
    print("DAVINCI_BOOT " + json.dumps({"task": task, "capabilities": snapshot}), flush=True)

    tracer = Tracer(stream=True)  # emits DAVINCI_TRACE lines as it goes
    mode = PermissionMode(os.environ.get("DAVINCI_PERMISSION_MODE", mode_default))
    agent = DavinciAgent(
        registry=registry,
        permissions=PermissionEngine(mode=mode, risk_ceiling=Risk.MEDIUM),
        reasoner=reasoner,  # None -> engine default (HeuristicReasoner)
        executor=executor,
        tracer=tracer,
        budget=Budget(max_steps=int(os.environ.get("DAVINCI_MAX_STEPS", "20"))),
    )

    result = agent.run(objective, required_categories=required)
    print("DAVINCI_RESULT " + json.dumps(result.to_dict()), flush=True)
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
