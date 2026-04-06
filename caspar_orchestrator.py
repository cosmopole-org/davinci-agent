"""Davinci orchestration built on Caspar VMM host-function APIs.

Design goal:
- tools are pre-running worker containers managed outside Davinci,
- Davinci only orchestrates by signaling tool points (no runVm deployment),
- tools send result/status back through signal points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
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


def bootstrap_default_registry() -> DavinciToolRegistry:
    registry = DavinciToolRegistry()

    def spec(machine: str, image: str, container: str, tool_id: str) -> ToolContainerSpec:
        tool_root = f"tools/{tool_id}"
        return ToolContainerSpec(
            machine_id=machine,
            image_name=image,
            container_name=container,
            tool_root=tool_root,
            dockerfile_path=f"{tool_root}/Dockerfile",
        )

    git_vm = VMDescriptor(
        name="git-vm",
        tools=[
            ToolCapability(
                tool_id="git_tool",
                categories=["version_control", "deployment"],
                functions={
                    "version_control": "status_commit",
                    "deployment": "open_pr",
                },
                description="Unified git tool with multi-function routing.",
                vm_name="git-vm",
                container=spec("git-machine", "git-tools:latest", "tool-git", "git_tool"),
                request_point="tool::git::request",
                response_point="tool::git::response",
                risk_level="high",
            ),
        ],
    )

    web_vm = VMDescriptor(
        name="web-vm",
        tools=[
            ToolCapability(
                tool_id="web_search",
                categories=["research"],
                functions={"research": "search"},
                description="Search web",
                vm_name="web-vm",
                container=spec("web-machine", "web-tools:latest", "tool-web-search", "web_search"),
                request_point="tool::web_search::request",
                response_point="tool::web_search::response",
                risk_level="medium",
                requires_network=True,
            ),
            ToolCapability(
                tool_id="fetch_url",
                categories=["research"],
                functions={"research": "fetch"},
                description="Fetch URL content",
                vm_name="web-vm",
                container=spec("web-machine", "web-tools:latest", "tool-fetch-url", "fetch_url"),
                request_point="tool::fetch_url::request",
                response_point="tool::fetch_url::response",
                risk_level="medium",
                requires_network=True,
            ),
            ToolCapability(
                tool_id="browser_automation",
                categories=["ui_automation"],
                functions={"ui_automation": "automate"},
                description="Automate browser",
                vm_name="web-vm",
                container=spec(
                    "web-machine",
                    "browser-tools:latest",
                    "tool-browser-automation",
                    "browser_automation",
                ),
                request_point="tool::browser_automation::request",
                response_point="tool::browser_automation::response",
                risk_level="high",
                requires_network=True,
            ),
        ],
    )

    data_vm = VMDescriptor(
        name="data-vm",
        tools=[
            ToolCapability(
                tool_id="vector_search",
                categories=["knowledge_retrieval"],
                functions={"knowledge_retrieval": "vector_search"},
                description="Semantic retrieval",
                vm_name="data-vm",
                container=spec("data-machine", "data-tools:latest", "tool-vector-search", "vector_search"),
                request_point="tool::vector_search::request",
                response_point="tool::vector_search::response",
                risk_level="low",
            ),
            ToolCapability(
                tool_id="sql_query",
                categories=["analytics"],
                functions={"analytics": "query"},
                description="Run SQL",
                vm_name="data-vm",
                container=spec("data-machine", "data-tools:latest", "tool-sql-query", "sql_query"),
                request_point="tool::sql_query::request",
                response_point="tool::sql_query::response",
                risk_level="high",
            ),
            ToolCapability(
                tool_id="python_exec",
                categories=["computation"],
                functions={"computation": "execute"},
                description="Run Python jobs",
                vm_name="data-vm",
                container=spec("data-machine", "data-tools:latest", "tool-python-exec", "python_exec"),
                request_point="tool::python_exec::request",
                response_point="tool::python_exec::response",
                risk_level="medium",
            ),
        ],
    )

    miniapps_vm = VMDescriptor(
        name="caspar-miniapps-vm",
        tools=[
            ToolCapability(
                tool_id="slack_connector",
                categories=["integrations"],
                functions={"integrations": "slack_action"},
                description="Slack actions",
                vm_name="caspar-miniapps-vm",
                container=spec("apps-machine", "apps-tools:latest", "tool-slack-connector", "slack_connector"),
                request_point="tool::slack_connector::request",
                response_point="tool::slack_connector::response",
                risk_level="medium",
                requires_network=True,
            ),
            ToolCapability(
                tool_id="jira_connector",
                categories=["integrations"],
                functions={"integrations": "jira_action"},
                description="Jira actions",
                vm_name="caspar-miniapps-vm",
                container=spec("apps-machine", "apps-tools:latest", "tool-jira-connector", "jira_connector"),
                request_point="tool::jira_connector::request",
                response_point="tool::jira_connector::response",
                risk_level="high",
                requires_network=True,
            ),
            ToolCapability(
                tool_id="calendar_connector",
                categories=["integrations"],
                functions={"integrations": "calendar_action"},
                description="Calendar actions",
                vm_name="caspar-miniapps-vm",
                container=spec(
                    "apps-machine",
                    "apps-tools:latest",
                    "tool-calendar-connector",
                    "calendar_connector",
                ),
                request_point="tool::calendar_connector::request",
                response_point="tool::calendar_connector::response",
                risk_level="high",
                requires_network=True,
            ),
        ],
    )

    for vm in (git_vm, web_vm, data_vm, miniapps_vm):
        registry.register_vm(vm)

    return registry


def summarize_tool_coverage(registry: DavinciToolRegistry) -> Dict[str, Iterable[str]]:
    categories: Dict[str, List[str]] = {}
    for t in registry.list_tools():
        for category in t.categories:
            categories.setdefault(category, []).append(
                f"{t.vm_name}:{t.tool_id}:{t.functions.get(category)}:{t.request_point}->{t.response_point}"
            )
    return categories
