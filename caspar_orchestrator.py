"""Davinci orchestration built on Caspar VMM host-function APIs.

Design goal:
- tools are pre-running worker containers managed outside Davinci,
- Davinci only orchestrates by signaling tool points (no runVm deployment),
- tools send result/status back through signal points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import socket
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class VmmHostFunction:
    SIGNAL_POINT: str = "signalPoint"
    HTTP_POST: str = "httpPost"
    RUN_VM: str = "runVm"
    EXEC_VM: str = "execVm"
    COPY_TO_VM: str = "copyToVm"
    CHECK_TOKEN_VALIDITY: str = "checkTokenValidity"
    TERMINATE_VM: str = "terminateVm"
    PLANT_TRIGGER: str = "plantTrigger"
    SEND_MESSAGE_ON_CHAIN: str = "sendMessageOnChain"
    LOG: str = "log"


@dataclass(frozen=True)
class ToolContainerSpec:
    """Container descriptor for a pre-running tool VM."""

    machine_id: str
    image_name: str
    container_name: str
    tool_root: str
    dockerfile_path: str
    vm_type: str = "docker"


@dataclass(frozen=True)
class ToolCapability:
    """A tool VM with multiple category->function routes."""

    tool_id: str
    categories: List[str]
    functions: Dict[str, str]
    description: str
    vm_name: str
    container: ToolContainerSpec
    request_point: str
    response_point: str
    risk_level: str = "medium"
    requires_network: bool = False
    prompt: Optional[str] = None
    function_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class VMDescriptor:
    name: str
    tools: List[ToolCapability] = field(default_factory=list)


@dataclass(frozen=True)
class ActionRequest:
    task: str
    required_categories: List[str]
    risk_tolerance: str = "medium"


@dataclass(frozen=True)
class ExecutionAction:
    """One concrete signaling action routed through Caspar host functions."""

    phase: str  # signal_request
    tool_id: str
    vm_name: str
    category: str
    host_function: str
    input_payload: Dict[str, Any]
    reason: str


@dataclass(frozen=True)
class ExecutionPlan:
    task: str
    actions: List[ExecutionAction]
    human_review_required: bool
    rationale: str


@dataclass(frozen=True)
class WorkerTemplate:
    """Blueprint for a short-lived worker VM managed by the master agent."""

    role: str
    image_name: str
    vm_name: str
    request_point: str
    response_point: str
    categories: List[str]
    startup_command: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    vm_type: str = "docker"
    risk_level: str = "medium"


@dataclass(frozen=True)
class WorkerInstance:
    """One runtime worker VM created for a specific parent task."""

    worker_id: str
    role: str
    vm_name: str
    machine_id: str
    request_point: str
    response_point: str
    task_id: str
    categories: List[str]
    risk_level: str = "medium"


class CasparVmmTransport:
    """Packet transport compatible with Caspar Docker SDK framing."""

    def __init__(self, host: str = "10.10.0.3", port: int = 8084, timeout_sec: int = 10) -> None:
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec

    def send(self, payload: Dict[str, Any], callback_id: int = 1) -> None:
        body = json.dumps(payload).encode("utf-8")
        length = len(body).to_bytes(4, byteorder="little", signed=False)
        cb = int(callback_id).to_bytes(8, byteorder="little", signed=False)
        packet = length + cb + body

        with socket.create_connection((self.host, self.port), timeout=self.timeout_sec) as conn:
            conn.sendall(packet)


class CasparVmmClient:
    """Client that emits Caspar host-call packets for worker VM interactions."""

    def __init__(self, transport: Optional[CasparVmmTransport] = None) -> None:
        self.transport = transport or CasparVmmTransport()

    def host_call(self, key: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"key": key, "input": input_data}
        self.transport.send(payload=payload, callback_id=1)
        return {"ok": True, "key": key, "input": input_data}

    def signal_tool(
        self,
        tool: ToolCapability,
        category: str,
        function_name: str,
        task: str,
        user_id: str = "davinci-agent",
    ) -> Dict[str, Any]:
        data = {
            "toolId": tool.tool_id,
            "machineId": tool.container.machine_id,
            "containerName": tool.container.container_name,
            "category": category,
            "function": function_name,
            "task": task,
            "expectedResponsePoint": tool.response_point,
        }
        return self.host_call(
            VmmHostFunction.SIGNAL_POINT,
            {
                "type": "broadcast",
                "pointId": tool.request_point,
                "userId": user_id,
                "data": json.dumps(data),
            },
        )

    def execute_plan(self, plan: ExecutionPlan) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for action in plan.actions:
            results.append(self.host_call(action.host_function, action.input_payload))
        return results

    def list_point_apps(self, point_id: str) -> Dict[str, Any]:
        """Fetch installed machines/apps/programs for a point via Caspar protocol API."""
        result = self.host_call(
            VmmHostFunction.HTTP_POST,
            {
                "path": "/points/listApps",
                "data": json.dumps({"pointId": point_id}),
                "headers": {"content-type": "application/json"},
            },
        )
        if isinstance(result, dict) and isinstance(result.get("response"), dict):
            return result["response"]
        return {}


class DavinciToolRegistry:
    def __init__(self) -> None:
        self._vms: Dict[str, VMDescriptor] = {}

    def register_vm(self, vm: VMDescriptor) -> None:
        self._vms[vm.name] = vm

    def list_vms(self) -> List[VMDescriptor]:
        return list(self._vms.values())

    def list_tools(self) -> List[ToolCapability]:
        tools: List[ToolCapability] = []
        for vm in self._vms.values():
            tools.extend(vm.tools)
        return tools


class MasterWorkerOrchestrator:
    """HiClaw-style manager/worker orchestration on top of Caspar host functions."""

    def __init__(self, client: CasparVmmClient, templates: Optional[List[WorkerTemplate]] = None) -> None:
        self.client = client
        self.templates: Dict[str, WorkerTemplate] = {template.role: template for template in templates or []}
        self._workers: Dict[str, WorkerInstance] = {}

    def register_template(self, template: WorkerTemplate) -> None:
        self.templates[template.role] = template

    def list_workers(self) -> List[WorkerInstance]:
        return list(self._workers.values())

    def launch_worker(self, role: str, task_id: str, worker_id: Optional[str] = None) -> WorkerInstance:
        template = self.templates[role]
        resolved_worker_id = worker_id or f"{role}-{task_id}"
        machine_id = f"{template.vm_name}-{resolved_worker_id}"

        run_payload = {
            "name": machine_id,
            "type": template.vm_type,
            "imageName": template.image_name,
            "cmd": template.startup_command or "",
            "envs": template.env,
            "metadata": {
                "role": template.role,
                "taskId": task_id,
                "requestPoint": template.request_point,
                "responsePoint": template.response_point,
            },
        }
        self.client.host_call(VmmHostFunction.RUN_VM, run_payload)

        worker = WorkerInstance(
            worker_id=resolved_worker_id,
            role=template.role,
            vm_name=template.vm_name,
            machine_id=machine_id,
            request_point=template.request_point,
            response_point=template.response_point,
            task_id=task_id,
            categories=list(template.categories),
            risk_level=template.risk_level,
        )
        self._workers[worker.worker_id] = worker
        return worker

    def assign_task(
        self,
        worker_id: str,
        task: str,
        category: str,
        function_name: str = "invoke",
        user_id: str = "davinci-master",
    ) -> Dict[str, Any]:
        worker = self._workers[worker_id]
        data = {
            "workerId": worker.worker_id,
            "machineId": worker.machine_id,
            "role": worker.role,
            "taskId": worker.task_id,
            "category": category,
            "function": function_name,
            "task": task,
            "expectedResponsePoint": worker.response_point,
        }
        return self.client.host_call(
            VmmHostFunction.SIGNAL_POINT,
            {
                "type": "broadcast",
                "pointId": worker.request_point,
                "userId": user_id,
                "data": json.dumps(data),
            },
        )

    def terminate_worker(self, worker_id: str) -> Dict[str, Any]:
        worker = self._workers.pop(worker_id)
        return self.client.host_call(VmmHostFunction.TERMINATE_VM, {"machineId": worker.machine_id})

    def plan_hierarchical_execution(self, task: str, required_categories: List[str]) -> ExecutionPlan:
        actions: List[ExecutionAction] = []
        missing: List[str] = []
        high_risk = False
        task_id = task.lower().replace(" ", "-")[:48] or "task"

        # launch one worker per template needed for category coverage
        launched: Dict[str, WorkerInstance] = {}
        for category in required_categories:
            template = next((t for t in self.templates.values() if category in t.categories), None)
            if not template:
                missing.append(category)
                continue
            if template.role not in launched:
                worker = self.launch_worker(role=template.role, task_id=task_id)
                launched[template.role] = worker
                if template.risk_level == "high":
                    high_risk = True
                actions.append(
                    ExecutionAction(
                        phase="provision_worker",
                        tool_id=template.role,
                        vm_name=template.vm_name,
                        category=category,
                        host_function=VmmHostFunction.RUN_VM,
                        input_payload={
                            "name": worker.machine_id,
                            "type": template.vm_type,
                            "imageName": template.image_name,
                        },
                        reason=f"Provision worker role {template.role} for categories {template.categories}.",
                    )
                )

            worker = launched[template.role]
            actions.append(
                ExecutionAction(
                    phase="signal_request",
                    tool_id=worker.worker_id,
                    vm_name=worker.vm_name,
                    category=category,
                    host_function=VmmHostFunction.SIGNAL_POINT,
                    input_payload={
                        "type": "broadcast",
                        "pointId": worker.request_point,
                        "userId": "davinci-master",
                        "data": json.dumps(
                            {
                                "workerId": worker.worker_id,
                                "taskId": worker.task_id,
                                "category": category,
                                "task": task,
                                "expectedResponsePoint": worker.response_point,
                            }
                        ),
                    },
                    reason=f"Delegate {category} to worker role {worker.role}.",
                )
            )

        for worker in launched.values():
            actions.append(
                ExecutionAction(
                    phase="cleanup_worker",
                    tool_id=worker.worker_id,
                    vm_name=worker.vm_name,
                    category="lifecycle",
                    host_function=VmmHostFunction.TERMINATE_VM,
                    input_payload={"machineId": worker.machine_id},
                    reason=f"Terminate ephemeral worker {worker.worker_id} after task completion.",
                )
            )

        rationale = (
            "Master-worker orchestration plan generated with lifecycle control via runVm/terminateVm."
            if not missing
            else f"Missing worker templates for categories: {', '.join(missing)}"
        )
        return ExecutionPlan(
            task=task,
            actions=actions,
            human_review_required=high_risk or bool(missing),
            rationale=rationale,
        )


def _iter_entities(point_catalog: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for group in ("machines", "apps", "programs"):
        values = point_catalog.get(group, {})
        if isinstance(values, dict):
            yield from values.values()
        elif isinstance(values, list):
            yield from values


def _tool_arg_spec_to_schema(args: Any) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    required: List[str] = []
    if not isinstance(args, dict):
        return {"type": "object", "properties": properties}
    type_map = {
        "STRING": "string",
        "NUMBER": "number",
        "INTEGER": "integer",
        "BOOLEAN": "boolean",
        "ARRAY": "array",
        "OBJECT": "object",
    }
    for arg_name, arg_meta in args.items():
        if not isinstance(arg_meta, dict):
            continue
        raw_type = str(arg_meta.get("type", "STRING")).upper()
        properties[arg_name] = {
            "type": type_map.get(raw_type, "string"),
            "description": str(arg_meta.get("desc", "")),
        }
        required.append(arg_name)
    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _capability_from_entity_metadata(entity: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[ToolCapability]:
    if not isinstance(metadata.get("tools"), list):
        return None
    tools = [t for t in metadata["tools"] if isinstance(t, dict)]
    if not tools:
        return None

    # MCP-style metadata (`isMcp: true`, `tools: [{name, desc, args}]`)
    if metadata.get("isMcp") is True and all("name" in t for t in tools):
        tool_id = str(metadata.get("tool_id") or entity.get("username") or entity.get("userId") or "mcp_tool")
        vm_name = str(metadata.get("vm_name", "mcp-vm"))
        request_point = str(metadata.get("request_point", f"tool::{tool_id}::request"))
        response_point = str(metadata.get("response_point", f"tool::{tool_id}::response"))
        categories = metadata.get("categories", ["mcp"])
        if not isinstance(categories, list) or not categories:
            categories = ["mcp"]

        routes = metadata.get("routes")
        functions: Dict[str, str]
        if isinstance(routes, dict):
            functions = {str(k): str(v) for k, v in routes.items()}
        else:
            default_function = str(tools[0].get("name", "invoke"))
            functions = {str(category): default_function for category in categories}

        tool_index = {str(t.get("name")): t for t in tools}
        function_schemas: Dict[str, Dict[str, Any]] = {}
        for category, fn_name in functions.items():
            fn_meta = tool_index.get(fn_name, {})
            function_schemas[category] = {
                "name": fn_name,
                "description": str(fn_meta.get("desc", "")),
                "input_schema": _tool_arg_spec_to_schema(fn_meta.get("args", {})),
                "output_schema": fn_meta.get("output_schema", {"type": "object"}),
            }

        return ToolCapability(
            tool_id=tool_id,
            categories=[str(c) for c in categories],
            functions=functions,
            description=str(metadata.get("description", metadata.get("desc", tool_id))),
            vm_name=vm_name,
            container=ToolContainerSpec(
                machine_id=str(metadata.get("machine_id") or entity.get("userId", "")),
                image_name=str(metadata.get("image_name", "")),
                container_name=str(metadata.get("container_name", "")),
                tool_root=f"tools/{tool_id}",
                dockerfile_path=f"tools/{tool_id}/Dockerfile",
                vm_type=str(metadata.get("vm_type", "docker")),
            ),
            request_point=request_point,
            response_point=response_point,
            risk_level=str(metadata.get("risk_level", "medium")),
            requires_network=bool(metadata.get("requires_network", False)),
            prompt=metadata.get("prompt"),
            function_schemas=function_schemas,
        )
    return None


def discover_tool_capabilities(point_catalog: Dict[str, Any]) -> List[ToolCapability]:
    """Load tool capabilities from installed machines/apps/programs metadata."""

    capabilities: List[ToolCapability] = []
    for entity in _iter_entities(point_catalog):
        if not isinstance(entity, dict):
            continue
        metadata = entity.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        mcp_capability = _capability_from_entity_metadata(entity, metadata)
        if mcp_capability:
            capabilities.append(mcp_capability)
            continue
        tools = metadata.get("tools", [])
        if not isinstance(tools, list):
            continue
        for raw_tool in tools:
            if not isinstance(raw_tool, dict):
                continue
            categories = raw_tool.get("categories")
            functions = raw_tool.get("functions")
            function_schemas: Dict[str, Dict[str, Any]] = {}

            if isinstance(functions, list):
                categories = []
                category_to_function: Dict[str, str] = {}
                for fn in functions:
                    if not isinstance(fn, dict):
                        continue
                    category = str(fn.get("category", "")).strip()
                    name = str(fn.get("name", "invoke")).strip() or "invoke"
                    if not category:
                        continue
                    if category not in categories:
                        categories.append(category)
                    category_to_function[category] = name
                    function_schemas[category] = {
                        "name": name,
                        "description": str(fn.get("description", "")),
                        "input_schema": fn.get("input_schema", {}),
                        "output_schema": fn.get("output_schema", {}),
                    }
                functions = category_to_function

            required = (
                "tool_id",
                "vm_name",
                "description",
                "request_point",
                "response_point",
            )
            if not all(key in raw_tool for key in required):
                continue
            if not isinstance(categories, list) or not isinstance(functions, dict):
                continue
            tool_id = str(raw_tool["tool_id"])
            tool_root = f"tools/{tool_id}"
            capabilities.append(
                ToolCapability(
                    tool_id=tool_id,
                    categories=list(categories),
                    functions=dict(functions),
                    description=str(raw_tool["description"]),
                    vm_name=str(raw_tool["vm_name"]),
                    container=ToolContainerSpec(
                        machine_id=str(raw_tool.get("machine_id") or entity.get("userId", "")),
                        image_name=str(raw_tool.get("image_name", "")),
                        container_name=str(raw_tool.get("container_name", "")),
                        tool_root=tool_root,
                        dockerfile_path=f"{tool_root}/Dockerfile",
                        vm_type=str(raw_tool.get("vm_type", "docker")),
                    ),
                    request_point=str(raw_tool["request_point"]),
                    response_point=str(raw_tool["response_point"]),
                    risk_level=str(raw_tool.get("risk_level", "medium")),
                    requires_network=bool(raw_tool.get("requires_network", False)),
                    prompt=raw_tool.get("prompt"),
                    function_schemas=function_schemas,
                )
            )
    return capabilities


class DavinciPlanner:
    """Plan by signaling already-running tool VMs (no deployment phase)."""

    def __init__(self, registry: DavinciToolRegistry) -> None:
        self.registry = registry

    def plan(self, req: ActionRequest) -> ExecutionPlan:
        tools_by_category: Dict[str, List[ToolCapability]] = {}
        for tool in self.registry.list_tools():
            for category in tool.categories:
                tools_by_category.setdefault(category, []).append(tool)

        actions: List[ExecutionAction] = []
        missing: List[str] = []
        high_risk = False

        for category in req.required_categories:
            options = tools_by_category.get(category, [])
            if not options:
                missing.append(category)
                continue

            tool = sorted(options, key=lambda t: (self._risk_rank(t.risk_level), t.tool_id))[0]
            if tool.risk_level == "high":
                high_risk = True

            function_name = tool.functions.get(category, "invoke")
            signal_payload = {
                "type": "broadcast",
                "pointId": tool.request_point,
                "userId": "davinci-agent",
                "data": json.dumps(
                    {
                        "task": req.task,
                        "toolId": tool.tool_id,
                        "category": category,
                        "function": function_name,
                        "containerName": tool.container.container_name,
                        "expectedResponsePoint": tool.response_point,
                    }
                ),
            }
            actions.append(
                ExecutionAction(
                    phase="signal_request",
                    tool_id=tool.tool_id,
                    vm_name=tool.vm_name,
                    category=category,
                    host_function=VmmHostFunction.SIGNAL_POINT,
                    input_payload=signal_payload,
                    reason=(
                        f"Signal pre-running tool {tool.tool_id} ({function_name}) and wait for "
                        f"response on {tool.response_point}."
                    ),
                )
            )

        rationale = "All required categories mapped to pre-running tools via point signaling."
        if missing:
            rationale = f"Missing capabilities: {', '.join(missing)}"

        return ExecutionPlan(
            task=req.task,
            actions=actions,
            human_review_required=high_risk or bool(missing) or req.risk_tolerance == "low",
            rationale=rationale,
        )

    @staticmethod
    def _risk_rank(level: str) -> int:
        return {"low": 0, "medium": 1, "high": 2}.get(level, 99)


def _load_catalog_from_env() -> Dict[str, Any]:
    raw = os.environ.get("CASPAR_POINT_APPS_JSON", "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def bootstrap_default_registry(
    point_id: Optional[str] = None,
    client: Optional[CasparVmmClient] = None,
    point_catalog: Optional[Dict[str, Any]] = None,
) -> DavinciToolRegistry:
    registry = DavinciToolRegistry()
    resolved_catalog = point_catalog
    if resolved_catalog is None:
        resolved_catalog = _load_catalog_from_env()
    if not resolved_catalog and point_id:
        resolved_catalog = (client or CasparVmmClient()).list_point_apps(point_id)

    capabilities = discover_tool_capabilities(resolved_catalog or {})
    grouped: Dict[str, List[ToolCapability]] = {}
    for capability in capabilities:
        grouped.setdefault(capability.vm_name, []).append(capability)

    for vm_name in sorted(grouped):
        vm_tools = sorted(grouped[vm_name], key=lambda t: t.tool_id)
        registry.register_vm(VMDescriptor(name=vm_name, tools=vm_tools))

    return registry


def summarize_tool_coverage(registry: DavinciToolRegistry) -> Dict[str, Iterable[str]]:
    categories: Dict[str, List[str]] = {}
    for t in registry.list_tools():
        for category in t.categories:
            categories.setdefault(category, []).append(
                f"{t.vm_name}:{t.tool_id}:{t.functions.get(category)}:{t.request_point}->{t.response_point}"
            )
    return categories
