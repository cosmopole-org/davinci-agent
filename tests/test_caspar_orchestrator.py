from pathlib import Path
import json

from caspar_orchestrator import (
    ActionRequest,
    CasparVmmClient,
    DavinciPlanner,
    MasterWorkerOrchestrator,
    VmmHostFunction,
    WorkerTemplate,
    bootstrap_default_registry,
)


class FakeTransport:
    def __init__(self):
        self.sent = []

    def send(self, payload, callback_id=1):
        self.sent.append((payload, callback_id))


def sample_point_catalog():
    return {
        "machines": {
            "git": {
                "userId": "git-machine",
                "metadata": {
                    "isMcp": True,
                    "tool_id": "git_tool",
                    "vm_name": "git-vm",
                    "machine_id": "git-machine",
                    "image_name": "git-tools:latest",
                    "container_name": "tool-git",
                    "categories": ["version_control", "deployment"],
                    "routes": {"version_control": "status_commit", "deployment": "open_pr"},
                    "description": "Unified git tool with multi-function routing.",
                    "prompt": "Use for status, commit, and PR automation.",
                    "request_point": "tool::git::request",
                    "response_point": "tool::git::response",
                    "risk_level": "high",
                    "tools": [
                        {
                            "name": "status_commit",
                            "desc": "Inspect repo status and create commits.",
                            "args": {"action": {"type": "STRING", "desc": "git action"}},
                        },
                        {
                            "name": "open_pr",
                            "desc": "Open a pull request.",
                            "args": {"title": {"type": "STRING", "desc": "pr title"}},
                        }
                    ]
                },
            },
            "web": {
                "userId": "web-machine",
                "metadata": {
                    "isMcp": True,
                    "tool_id": "web_search",
                    "vm_name": "web-vm",
                    "machine_id": "web-machine",
                    "image_name": "web-tools:latest",
                    "container_name": "tool-web-search",
                    "categories": ["research"],
                    "routes": {"research": "search"},
                    "description": "Search web",
                    "request_point": "tool::web_search::request",
                    "response_point": "tool::web_search::response",
                    "risk_level": "medium",
                    "requires_network": True,
                    "tools": [
                        {
                            "name": "search",
                            "desc": "Search the web.",
                            "args": {"query": {"type": "STRING", "desc": "query string"}},
                        }
                    ]
                },
            },
        },
        "apps": {
            "data": {
                "metadata": {
                    "isMcp": True,
                    "tool_id": "vector_search",
                    "vm_name": "data-vm",
                    "machine_id": "data-machine",
                    "image_name": "data-tools:latest",
                    "container_name": "tool-vector-search",
                    "categories": ["knowledge_retrieval"],
                    "routes": {"knowledge_retrieval": "vector_search"},
                    "description": "Semantic retrieval",
                    "request_point": "tool::vector_search::request",
                    "response_point": "tool::vector_search::response",
                    "risk_level": "low",
                    "tools": [
                        {
                            "name": "vector_search",
                            "desc": "Semantic vector query.",
                            "args": {"query": {"type": "STRING", "desc": "query text"}},
                        }
                    ]
                }
            }
        },
        "programs": [
            {
                "metadata": {
                    "isMcp": True,
                    "tool_id": "slack_connector",
                    "vm_name": "caspar-miniapps-vm",
                    "machine_id": "apps-machine",
                    "image_name": "apps-tools:latest",
                    "container_name": "tool-slack-connector",
                    "categories": ["integrations"],
                    "routes": {"integrations": "slack_action"},
                    "description": "Slack actions",
                    "request_point": "tool::slack_connector::request",
                    "response_point": "tool::slack_connector::response",
                    "risk_level": "medium",
                    "requires_network": True,
                    "tools": [
                        {
                            "name": "slack_action",
                            "desc": "Perform slack operation.",
                            "args": {"operation": {"type": "STRING", "desc": "operation"}},
                        }
                    ]
                }
            }
        ],
    }


def test_registry_uses_point_catalog_instead_of_local_tool_folder():
    registry = bootstrap_default_registry(point_catalog=sample_point_catalog())

    tools = registry.list_tools()
    git_tools = [tool for tool in tools if tool.vm_name == "git-vm"]
    assert len(git_tools) == 1
    assert git_tools[0].tool_id == "git_tool"
    assert set(git_tools[0].categories) == {"version_control", "deployment"}
    assert git_tools[0].functions["deployment"] == "open_pr"
    assert git_tools[0].prompt == "Use for status, commit, and PR automation."
    assert "version_control" in git_tools[0].function_schemas

    for tool in tools:
        assert tool.container.tool_root == f"tools/{tool.tool_id}"
        assert Path(tool.container.dockerfile_path).is_file()


def test_planner_uses_signal_only_for_pre_running_tools():
    registry = bootstrap_default_registry(point_catalog=sample_point_catalog())
    planner = DavinciPlanner(registry)

    req = ActionRequest(
        task="Research issue, update code, and open PR",
        required_categories=["research", "version_control", "deployment"],
        risk_tolerance="medium",
    )
    plan = planner.plan(req)

    assert len(plan.actions) == 3
    assert all(a.phase == "signal_request" for a in plan.actions)
    assert all(a.host_function == VmmHostFunction.SIGNAL for a in plan.actions)
    assert all("expectedResponsePoint" in a.input_payload["data"] for a in plan.actions)
    assert plan.human_review_required is True


def test_vmm_client_executes_plan_actions_via_signal():
    fake = FakeTransport()
    client = CasparVmmClient(transport=fake)

    registry = bootstrap_default_registry(point_catalog=sample_point_catalog())
    planner = DavinciPlanner(registry)
    plan = planner.plan(ActionRequest(task="quick research", required_categories=["research"]))

    results = client.execute_plan(plan)

    assert len(results) == 1
    assert fake.sent[0][0]["key"] == VmmHostFunction.SIGNAL
    assert fake.sent[0][0]["input"]["type"] == "machine"
    assert "machineId" in fake.sent[0][0]["input"]


def test_point_catalog_fetch_uses_protocol_api_host_function():
    fake = FakeTransport()
    client = CasparVmmClient(transport=fake)

    client.list_point_apps("point-123")

    assert fake.sent[0][0]["key"] == VmmHostFunction.PROTOCOL_API
    assert fake.sent[0][0]["input"]["apiKey"] == "/points/listApps"


def test_tool_point_metadata_files_include_function_schemas():
    for metadata_file in Path("tools").glob("*/point.metadata.json"):
        payload = json.loads(metadata_file.read_text())
        assert payload.get("isMcp") is True
        assert "tool_id" in payload
        assert "prompt" in payload
        assert isinstance(payload.get("tools"), list)
        assert payload["tools"]
        for fn in payload["tools"]:
            assert "name" in fn
            assert "desc" in fn
            assert "args" in fn


def test_master_worker_orchestrator_lifecycle_calls_run_and_terminate_vm():
    fake = FakeTransport()
    client = CasparVmmClient(transport=fake)
    orchestrator = MasterWorkerOrchestrator(
        client=client,
        templates=[
            WorkerTemplate(
                role="researcher",
                image_name="worker-research:latest",
                vm_name="research-worker-vm",
                request_point="worker::research::request",
                response_point="worker::research::response",
                categories=["research"],
            )
        ],
    )

    worker = orchestrator.launch_worker(role="researcher", task_id="task-123")
    assert worker.role == "researcher"
    assert fake.sent[0][0]["key"] == VmmHostFunction.RUN_VM

    orchestrator.assign_task(worker_id=worker.worker_id, task="find recent updates", category="research")
    assert fake.sent[1][0]["key"] == VmmHostFunction.SIGNAL

    orchestrator.terminate_worker(worker_id=worker.worker_id)
    assert fake.sent[2][0]["key"] == VmmHostFunction.TERMINATE_VM


def test_hierarchical_execution_plan_includes_worker_lifecycle_actions():
    fake = FakeTransport()
    client = CasparVmmClient(transport=fake)
    orchestrator = MasterWorkerOrchestrator(
        client=client,
        templates=[
            WorkerTemplate(
                role="coder",
                image_name="worker-coder:latest",
                vm_name="coder-worker-vm",
                request_point="worker::coder::request",
                response_point="worker::coder::response",
                categories=["implementation", "testing"],
                risk_level="high",
            )
        ],
    )

    plan = orchestrator.plan_hierarchical_execution(
        task="implement tests for module",
        required_categories=["implementation", "testing"],
    )

    phases = [action.phase for action in plan.actions]
    assert phases[0] == "provision_worker"
    assert phases.count("signal_request") == 2
    assert phases[-1] == "cleanup_worker"
    assert plan.human_review_required is True
