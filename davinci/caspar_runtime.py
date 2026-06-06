"""Docker-creature entrypoint for Davinci on a Caspar node.

When Davinci is deployed as a ``docker`` creature and run via
``/programs/runEntity``, the Caspar node:

  * uploads any ``params`` to ``/app/input`` inside the container, and
  * captures the container's stdout as VM logs (readable with
    ``/machines/readVmLogs``).

This entrypoint therefore reads its task from ``/app/input`` (or the
``DAVINCI_TASK`` env var), runs the full Davinci agent loop with a streaming
tracer, and prints clearly-delimited, greppable JSON so a supervising client
can assert on the outcome over the signalling API.

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


class CasparCreatureExecutor:
    """Live executor: Davinci signals sibling *tool creatures* via the node API.

    For each tool action the engine approves, this runs the tool creature's
    deployed entity (``/programs/runEntity`` — the signalling primitive) with the
    task payload, then reads the tool's VM logs back over the same API and
    extracts its ``TOOL_RESPONSE`` line. This is Davinci interacting with other
    creatures purely through Caspar's signalling surface.
    """

    def __init__(self, client: Any, tools_by_name: Dict[str, Dict[str, Any]]) -> None:
        self.client = client
        self.tools = tools_by_name

    def execute(self, tool: ToolDescriptor, action: ToolAction) -> ToolResult:
        spec = self.tools.get(tool.name)
        if not spec:
            return ToolResult(ok=False, output=None, error=f"unknown tool creature {tool.name}")
        # Send the reasoner's schema-correct arguments straight through as the
        # tool payload (the reasoner fills them from the tool's input_schema).
        # `task`/`step` are kept as context for tools that fall back to them.
        payload = {k: v for k, v in (action.args or {}).items() if v is not None}
        payload.setdefault("task", action.args.get("task", ""))
        payload.setdefault("step", action.args.get("step", ""))
        signal = {"tool_id": spec.get("tool_id", tool.name),
                  "function": spec.get("function", "invoke"),
                  "payload": payload}
        try:
            vm_id = self.client.run_entity(
                spec["program_id"], spec["entity_id"],
                params={"task.json": json.dumps(signal)},
                ram_mb=int(spec.get("ram_mb", 256)), max_exec_seconds=int(spec.get("max_exec_seconds", 60)))
            found, logs = self.client.wait_for_vm_log(vm_id, "TOOL_RESPONSE", timeout=75)
        except Exception as exc:
            return ToolResult(ok=False, output=None, error=f"signal failed: {exc}")
        from .caspar_signaling import _log_text
        response = None
        for entry in logs:
            text = _log_text(entry)
            if "TOOL_RESPONSE" in text:
                try:
                    response = json.loads(text.split("TOOL_RESPONSE", 1)[1].strip())
                except Exception:
                    response = {"raw": text}
        if not found:
            return ToolResult(ok=False, output={"vm_id": vm_id, "logs_tail": [_log_text(l) for l in logs[-5:]]},
                              error="no TOOL_RESPONSE from tool creature")
        return ToolResult(ok=True, output={"tool": tool.name, "vm_id": vm_id, "response": response})


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


def _discover_via_store(client: Any, store_cfg: Dict[str, Any]):
    """Path A discovery — list a store's machines, keep the ``isMcp`` ones, and
    build a **schema-rich** registry from each tool's MCP manifest.

    Davinci signals the Decillion *stores* miniapp creature's ``listMachines``
    action; the node's host function returns each machine with its metadata
    (the point manifest: ``isMcp``, ``categories``, ``routes``, ``tools[].args``).
    We feed those manifests through ``discover_tool_capabilities`` so every tool's
    function input-schema is materialised (deferred), then expose them as an MCP
    ``ToolRegistry``. Returns ``(registry, tools_by_name)``.
    """
    creature_id = store_cfg["creature_id"]
    program_id = store_cfg["program_id"]
    entity = store_cfg.get("entity", "main")
    store_id = store_cfg["store_id"]
    resp = client.signal_miniapp(
        creature_id=creature_id, program_id=program_id, entity=entity,
        action="listMachines", payload={"storeId": store_id, "mcpOnly": True},
        store_id=store_id, timeout=float(store_cfg.get("timeout", 40)))
    machines: List[Dict[str, Any]] = []
    if resp.get("ok"):
        machines = resp.get("result", {}).get("machines") or []

    catalog: Dict[str, Any] = {"machines": {}}
    tools_by_name: Dict[str, Dict[str, Any]] = {}
    mcp_count = 0
    for m in machines:
        meta = m.get("metadata") or {}
        is_mcp = bool(m.get("isMcp") or meta.get("isMcp"))
        if not is_mcp:
            continue
        mcp_count += 1
        program = m.get("programId") or meta.get("program_id") or ""
        tool_id = str(meta.get("tool_id") or program or f"tool{mcp_count}")
        name = f"caspar__{tool_id}"
        catalog["machines"][program or tool_id] = {"userId": program, "metadata": meta}
        cats = meta.get("categories") or ["general"]
        routes = meta.get("routes") or {}
        function = routes.get(cats[0]) or (meta.get("tools") or [{}])[0].get("name", "invoke")
        tools_by_name[name] = {
            "name": name, "tool_id": tool_id, "program_id": program,
            "entity_id": meta.get("entity_id") or tool_id, "function": function,
            "category": cats[0], "risk": meta.get("risk_level", "medium"),
            "requires_network": bool(meta.get("requires_network", False)),
        }

    from caspar_orchestrator import bootstrap_default_registry
    caspar_reg = bootstrap_default_registry(point_catalog=catalog)
    registry = ToolRegistry.from_caspar_registry(caspar_reg)
    print("DAVINCI_DISCOVERY " + json.dumps({
        "source": "store", "store_id": store_id, "listed": len(machines),
        "mcp": mcp_count, "registered": len(registry.all()),
        "tools": [t.name for t in registry.all()]}), flush=True)
    return registry, tools_by_name


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

    # Live mode: signal sibling tool creatures through the node. Configuration
    # (node address + tool catalog) is delivered as an input file by the signal
    # that started us; the creature then logs in *itself* as the owning admin to
    # obtain credentials — no private key is ever transmitted to the container.
    config = _read_config()
    tools = config.get("tools") or []
    store_cfg = config.get("store") or {}
    node_host = config.get("node_host") or os.environ.get("DAVINCI_NODE_HOST", "")
    username = config.get("username") or os.environ.get("DAVINCI_ADMIN_USERNAME", "")
    # Live when we can reach the node AND have a way to discover tools — either a
    # store to list (Path A) or a pre-resolved catalog.
    live = bool(node_host and username and (store_cfg or tools))

    executor: Any = EchoExecutor()
    discovery = "none"
    mode_default = "auto"

    # Gateway path: when the node injected CASPAR_GATEWAY_* env, this container
    # is a gateway-managed docker creature and the bridge is its *only* route to
    # the outside world. Use it for tool execution and skip the external client.
    bridge = None
    try:
        from .caspar_bridge import bridge_from_env
        bridge = bridge_from_env()
    except Exception as exc:  # noqa: BLE001 — fall back to legacy/offline paths
        print("DAVINCI_BOOT " + json.dumps({"bridge_init_error": repr(exc)[:160]}), flush=True)
        bridge = None

    if bridge is not None:
        my_id = os.environ.get("CASPAR_MACHINE_ID") or os.environ.get("CASPAR_PROGRAM_ID") or ""
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
    elif live:
        try:
            from .caspar_signaling import CasparSignalingClient
            client = CasparSignalingClient(node_host, int(config.get("node_port", 8074)))
            client.connect()
            client.login(username)  # idempotent dev login -> owning creds
            if store_cfg:
                # Path A: discover tools by listing the store's MCP machines.
                registry, tools_by_name = _discover_via_store(client, store_cfg)
                discovery = "store"
            else:
                # Fallback: a pre-resolved (schema-less) catalog from config.
                registry = _registry_from_tool_catalog(tools)
                tools_by_name = {t.get("name") or f"caspar__{t['tool_id']}": t for t in tools}
                discovery = "catalog"
            executor = CasparCreatureExecutor(client, tools_by_name)
            if not required:
                required = list(dict.fromkeys(
                    t.get("category", "general") for t in tools_by_name.values()))[:4]
            if not tools_by_name:
                raise RuntimeError("no tools discovered")
        except Exception as exc:
            print("DAVINCI_BOOT " + json.dumps({"live_mode_error": repr(exc)}), flush=True)
            registry = _build_registry()
            live = False
    else:
        registry = _build_registry()

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
