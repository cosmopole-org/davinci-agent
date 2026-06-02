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
        signal = {"tool_id": spec.get("tool_id", tool.name),
                  "function": spec.get("function", "invoke"),
                  "payload": {"task": action.args.get("task", ""),
                              "step": action.args.get("step", "")}}
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

    # Live mode: signal sibling tool creatures through the node. Configuration
    # (node address + tool catalog) is delivered as an input file by the signal
    # that started us; the creature then logs in *itself* as the owning admin to
    # obtain credentials — no private key is ever transmitted to the container.
    config = _read_config()
    tools = config.get("tools") or []
    node_host = config.get("node_host") or os.environ.get("DAVINCI_NODE_HOST", "")
    username = config.get("username") or os.environ.get("DAVINCI_ADMIN_USERNAME", "")
    live = bool(tools and node_host and username)

    executor: Any = EchoExecutor()
    mode_default = "auto"
    if live:
        try:
            registry = _registry_from_tool_catalog(tools)
            from .caspar_signaling import CasparSignalingClient
            client = CasparSignalingClient(node_host, int(config.get("node_port", 8074)))
            client.connect()
            client.login(username)  # idempotent dev login -> owning creds
            tools_by_name = {t.get("name") or f"caspar__{t['tool_id']}": t for t in tools}
            executor = CasparCreatureExecutor(client, tools_by_name)
            if not required:
                required = list(dict.fromkeys(t.get("category", "general") for t in tools))[:4]
        except Exception as exc:
            print("DAVINCI_BOOT " + json.dumps({"live_mode_error": repr(exc)}), flush=True)
            registry = _build_registry()
            live = False
    else:
        registry = _build_registry()

    snapshot = _capability_snapshot(registry)
    snapshot["live_signaling"] = live
    print("DAVINCI_BOOT " + json.dumps({"task": task, "capabilities": snapshot}), flush=True)

    tracer = Tracer(stream=True)  # emits DAVINCI_TRACE lines as it goes
    mode = PermissionMode(os.environ.get("DAVINCI_PERMISSION_MODE", mode_default))
    agent = DavinciAgent(
        registry=registry,
        permissions=PermissionEngine(mode=mode, risk_ceiling=Risk.MEDIUM),
        executor=executor,
        tracer=tracer,
        budget=Budget(max_steps=int(os.environ.get("DAVINCI_MAX_STEPS", "20"))),
    )

    result = agent.run(objective, required_categories=required)
    print("DAVINCI_RESULT " + json.dumps(result.to_dict()), flush=True)
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
