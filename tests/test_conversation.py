"""Tests for Davinci's conversation/session support."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davinci import DavinciAgent, EchoExecutor, SessionMemory  # noqa: E402
from davinci.engine import _normalize_history  # noqa: E402
from davinci.reasoner import LLMReasoner  # noqa: E402


# --------------------------------------------------------------------------- #
# history normalization
# --------------------------------------------------------------------------- #

def test_normalize_history_coerces_roles_and_shapes():
    convo = _normalize_history([
        {"role": "user", "content": "hi"},
        {"from": "agent", "text": "hello", "agentName": "Ada"},
        {"role": "assistant", "message": "second"},
        {"role": "system", "content": "  "},          # blank -> dropped
        "garbage",                                       # non-dict -> dropped
        {"sender": "human", "content": "again"},
    ])
    assert [m["role"] for m in convo] == ["user", "assistant", "assistant", "user"]
    assert convo[0]["content"] == "hi"
    assert convo[1]["agentName"] == "Ada"


def test_normalize_history_handles_none():
    assert _normalize_history(None) == []


# --------------------------------------------------------------------------- #
# engine seeds conversation
# --------------------------------------------------------------------------- #

def test_run_seeds_conversation_into_memory():
    agent = DavinciAgent(executor=EchoExecutor())
    history = [
        {"role": "user", "content": "what is our project name"},
        {"role": "agent", "content": "It's called Orbit."},
    ]
    result = agent.run("remind me what it's called", history=history)
    convo = agent.working.get_fact("conversation")
    assert convo and len(convo) == 2
    assert convo[1]["role"] == "assistant"
    # Seeded turns are visible in working memory alongside the run's own items.
    roles = [m["role"] for m in agent.working.items]
    assert "user" in roles and "assistant" in roles
    assert result.answer  # run still completes


def test_run_notifies_reasoner_set_conversation():
    seen = {}

    class Rec:
        def set_conversation(self, history):
            seen["history"] = history

        def propose(self, objective, step, registry, memory):
            from davinci.engine import ActionProposal
            return ActionProposal(tool=None, final_answer="done")

        def reflect(self, objective, plan, memory):
            from davinci.engine import Reflection
            return Reflection(satisfied=True)

    agent = DavinciAgent(reasoner=Rec(), executor=EchoExecutor())
    agent.run("go", history=[{"role": "user", "content": "prior"}])
    assert seen.get("history") and seen["history"][0]["content"] == "prior"


# --------------------------------------------------------------------------- #
# SessionMemory
# --------------------------------------------------------------------------- #

def test_session_memory_append_and_messages():
    mem = SessionMemory()
    mem.append("s1", "user", "hi")
    mem.append("s1", "assistant", "hello", agent_name="Ada")
    mem.append("s1", "user", "   ")  # blank ignored
    msgs = mem.messages("s1")
    assert len(msgs) == 2
    assert msgs[1]["agentName"] == "Ada"
    assert mem.messages("other") == []


def test_session_memory_replace_is_authoritative_and_bounded():
    mem = SessionMemory(max_messages=3)
    mem.append("s", "user", "old")
    mem.replace("s", [{"role": "user", "content": f"m{i}"} for i in range(5)])
    msgs = mem.messages("s")
    assert len(msgs) == 3  # bounded to max_messages
    assert msgs[0]["content"] == "m2"  # kept the most recent


def test_session_memory_evicts_least_recently_used():
    mem = SessionMemory(max_sessions=2)
    mem.append("a", "user", "a")
    mem.append("b", "user", "b")
    mem.append("a", "user", "a2")  # touch a -> b is now LRU
    mem.append("c", "user", "c")   # evicts b
    assert mem.messages("b") == []
    assert mem.messages("a") and mem.messages("c")


# --------------------------------------------------------------------------- #
# LLMReasoner threads conversation into its prompts
# --------------------------------------------------------------------------- #

class _CapturingClient:
    """Minimal LLMClient stand-in that records prompts and returns canned JSON."""

    provider = "fake"
    models = ["fake-1"]
    active_model = "fake-1"

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[tuple[str, str]] = []

    def generate(self, prompt, system="", attachments=None):
        self.prompts.append((system or "", prompt))
        return self.reply

    def emit(self, *a, **k):
        pass


def test_reasoner_includes_conversation_in_plan_and_synthesis():
    from davinci.mcp import ToolRegistry
    from davinci.memory import WorkingMemory
    from davinci.planning import Plan

    client = _CapturingClient('{"complexity": "trivial", "steps": [{"title": "answer", "category": "result"}]}')
    reasoner = LLMReasoner(client)
    reasoner.set_conversation([{"role": "user", "content": "call it Orbit"}])

    reasoner.make_plan("what did we name it", [], ToolRegistry())
    plan_system, plan_prompt = client.prompts[-1]
    assert "conversation" in plan_prompt.lower()
    assert "ongoing conversation" in plan_system.lower()

    # synthesize answers even with no tool results when a conversation exists.
    client.reply = '{"final_answer": "Orbit"}'
    plan = Plan(objective="what did we name it")
    out = reasoner.synthesize("what did we name it", plan, WorkingMemory())
    assert out == "Orbit"
    syn_system, syn_prompt = client.prompts[-1]
    assert "conversation" in syn_prompt.lower()


def test_reasoner_set_conversation_ignores_blank():
    client = _CapturingClient("{}")
    reasoner = LLMReasoner(client)
    reasoner.set_conversation([{"role": "user", "content": "  "}, {"role": "user", "content": "keep"}])
    assert reasoner._conversation == [{"role": "user", "content": "keep"}]
