"""Unit tests for the Davinci enterprise agent package (stdlib only)."""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davinci import (  # noqa: E402
    Budget, DavinciAgent, EchoExecutor, InstructionMemory, Outcome,
    PermissionEngine, PermissionMode, Risk, ToolAction, ToolDescriptor,
    ToolRegistry, Tracer, mask_secrets,
)
from davinci.engine import HeuristicReasoner, ToolResult  # noqa: E402
from davinci.memory import EpisodicMemory, WorkingMemory, compact_messages  # noqa: E402
from davinci.planning import Plan, Planner, StepStatus  # noqa: E402


# --------------------------------------------------------------------------- #
# observability
# --------------------------------------------------------------------------- #

def test_mask_secrets_redacts_keys_and_patterns():
    assert mask_secrets({"api_key": "abc123"})["api_key"] == "***REDACTED***"
    masked = mask_secrets("token=supersecretvalue123")
    assert "supersecretvalue123" not in masked
    assert mask_secrets("sk-" + "A" * 30).startswith("***") or "REDACTED" in mask_secrets("sk-" + "A" * 30)


def test_budget_limits():
    b = Budget(max_steps=2)
    assert b.exceeded() is None
    b.charge_step(); b.charge_step()
    assert b.exceeded() == "max_steps"


def test_tracer_is_append_only_and_masks():
    t = Tracer()
    t.emit("reason", "thinking", password="hunter2")
    events = t.events
    assert len(events) == 1
    assert events[0].data["password"] == "***REDACTED***"
    assert json.loads(t.to_jsonl())["kind"] == "reason"


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #

def test_planner_produces_ordered_plan():
    plan = Planner().plan("do the thing", ["web_research"])
    assert plan.steps
    assert plan.next_pending().id == plan.steps[0].id


def test_plan_lifecycle_and_replan():
    plan = Plan("obj")
    s1 = plan.add_step("a")
    plan.start(s1.id); plan.fail(s1.id, "boom")
    assert plan.has_failures
    plan.replan_failed(["a-retry"])
    assert plan.steps[0].status == StepStatus.SKIPPED
    assert any(s.title == "a-retry" for s in plan.steps)
    assert plan.revision == 1


# --------------------------------------------------------------------------- #
# permissions
# --------------------------------------------------------------------------- #

def test_permission_deny_first():
    eng = PermissionEngine(mode=PermissionMode.BYPASS, deny_rules=["*:danger_tool"])
    d = eng.check(ToolAction("danger_tool", "x", Risk.LOW))
    assert d.outcome == Outcome.DENY


def test_permission_default_reviews_medium():
    eng = PermissionEngine(mode=PermissionMode.DEFAULT)
    assert eng.check(ToolAction("t", "c", Risk.LOW)).outcome == Outcome.ALLOW
    assert eng.check(ToolAction("t", "c", Risk.MEDIUM)).outcome == Outcome.REVIEW


def test_dangerous_command_guardrail():
    eng = PermissionEngine(mode=PermissionMode.BYPASS)
    d = eng.check(ToolAction("sh", "exec", Risk.LOW, args={"cmd": "rm -rf /"}))
    assert d.outcome == Outcome.DENY


def test_output_guardrail_caps_size():
    eng = PermissionEngine()
    err = eng.validate_result(ToolAction("t", "c"), "x" * 6_000_000)
    assert err is not None


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #

def test_working_memory_bounded():
    wm = WorkingMemory(max_items=5)
    for i in range(20):
        wm.remember("user", i)
    assert len(wm.items) == 5


def test_episodic_replay_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "events.jsonl")
        mem = EpisodicMemory(path=path)
        mem.append({"kind": "a"}); mem.append({"kind": "b"})
        replayed = EpisodicMemory.replay(path)
        assert [e["kind"] for e in replayed.events] == ["a", "b"]


def test_instruction_memory_hierarchy(tmp_path=None):
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "proj")
        sub = os.path.join(root, "a", "b")
        os.makedirs(sub)
        open(os.path.join(root, "DAVINCI.md"), "w").write("root rule")
        open(os.path.join(sub, "DAVINCI.md"), "w").write("leaf rule")
        im = InstructionMemory().discover(sub, root_dir=root)
        rendered = im.render()
        assert "root rule" in rendered and "leaf rule" in rendered
        # leaf (more specific) appears after root
        assert rendered.index("leaf rule") > rendered.index("root rule")


def test_compact_messages_trims():
    msgs = [{"role": "user", "content": "x" * 10000}] + \
           [{"role": "a", "content": str(i)} for i in range(100)]
    out = compact_messages(msgs, max_chars_per_item=100, max_items=10)
    assert len(out) == 10
    assert len(out[0]["content"]) <= 200


# --------------------------------------------------------------------------- #
# mcp registry
# --------------------------------------------------------------------------- #

def test_registry_search_and_deferred_schema():
    reg = ToolRegistry()
    loaded = {"n": 0}

    def loader():
        loaded["n"] += 1
        return {"type": "object", "properties": {"q": {"type": "string"}}}

    reg.register(ToolDescriptor("local__web", "search the web", category="web_research",
                                risk="low", _schema_loader=loader))
    reg.register(ToolDescriptor("local__git", "git operations", category="version_control"))
    assert loaded["n"] == 0  # schema deferred until used
    hits = reg.search("web")
    assert hits and hits[0].name == "local__web"
    assert reg.search("select:local__git")[0].name == "local__git"
    _ = reg.get("local__web").schema()
    assert loaded["n"] == 1


# --------------------------------------------------------------------------- #
# engine end-to-end
# --------------------------------------------------------------------------- #

def _registry_with_tools():
    reg = ToolRegistry()
    reg.register(ToolDescriptor("local__web_search", "search", category="web_research", risk="low"))
    reg.register(ToolDescriptor("local__exec", "execute", category="execution", risk="medium"))
    return reg


def test_agent_runs_plan_to_completion():
    agent = DavinciAgent(
        registry=_registry_with_tools(),
        permissions=PermissionEngine(mode=PermissionMode.AUTO, risk_ceiling=Risk.MEDIUM),
        executor=EchoExecutor(),
        tracer=Tracer(),
        budget=Budget(max_steps=20),
    )
    result = agent.run("research and execute", required_categories=["web_research", "execution"])
    assert result.success
    assert result.plan["progress"]["done"] >= 1
    assert result.trace["event_count"] > 0
    assert result.budget["tool_calls_used"] >= 1


def test_agent_escalates_high_risk_for_review():
    reg = ToolRegistry()
    reg.register(ToolDescriptor("local__deploy", "prod deploy", category="execution", risk="high"))
    agent = DavinciAgent(
        registry=reg,
        permissions=PermissionEngine(mode=PermissionMode.DEFAULT),
        executor=EchoExecutor(),
        budget=Budget(max_steps=20),
    )
    result = agent.run("ship it", required_categories=["execution"])
    assert result.pending_review, "high-risk action must be escalated"


def test_agent_stuck_detection_fails_step():
    class LoopReasoner(HeuristicReasoner):
        def reflect(self, objective, plan, memory):
            from davinci.engine import Reflection
            return Reflection(satisfied=True)

    class FlakyExecutor:
        def execute(self, tool, action):
            return ToolResult(ok=False, output=None, error="always fails")

    reg = _registry_with_tools()
    agent = DavinciAgent(registry=reg, permissions=PermissionEngine(mode=PermissionMode.BYPASS),
                         reasoner=LoopReasoner(), executor=FlakyExecutor(),
                         budget=Budget(max_steps=30), max_step_attempts=1)
    result = agent.run("loop", required_categories=["execution"])
    # failing executor => not success, but the loop must terminate (no hang)
    assert result.success is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
