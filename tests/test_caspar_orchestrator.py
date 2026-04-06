from pathlib import Path

from caspar_orchestrator import (
    ActionRequest,
    CasparVmmClient,
    DavinciPlanner,
    VmmHostFunction,
    bootstrap_default_registry,
)


class FakeTransport:
    def __init__(self):
        self.sent = []

    def send(self, payload, callback_id=1):
        self.sent.append((payload, callback_id))


def test_registry_uses_unified_git_tool_and_isolated_tool_folders():
    registry = bootstrap_default_registry()

    tools = registry.list_tools()
    git_tools = [tool for tool in tools if tool.vm_name == "git-vm"]
    assert len(git_tools) == 1
    assert git_tools[0].tool_id == "git_tool"
    assert set(git_tools[0].categories) == {"version_control", "deployment"}

    for tool in tools:
        assert tool.container.tool_root == f"tools/{tool.tool_id}"
        assert Path(tool.container.dockerfile_path).is_file()


def test_planner_uses_signal_only_for_pre_running_tools():
    registry = bootstrap_default_registry()
    planner = DavinciPlanner(registry)

    req = ActionRequest(
        task="Research issue, update code, and open PR",
        required_categories=["research", "version_control", "deployment"],
        risk_tolerance="medium",
    )
    plan = planner.plan(req)

    assert len(plan.actions) == 3
    assert all(a.phase == "signal_request" for a in plan.actions)
    assert all(a.host_function == VmmHostFunction.SIGNAL_POINT for a in plan.actions)
    assert all("expectedResponsePoint" in a.input_payload["data"] for a in plan.actions)
    assert plan.human_review_required is True


def test_vmm_client_executes_plan_actions_via_signal_point():
    fake = FakeTransport()
    client = CasparVmmClient(transport=fake)

    registry = bootstrap_default_registry()
    planner = DavinciPlanner(registry)
    plan = planner.plan(ActionRequest(task="quick research", required_categories=["research"]))

    results = client.execute_plan(plan)

    assert len(results) == 1
    assert fake.sent[0][0]["key"] == VmmHostFunction.SIGNAL_POINT
