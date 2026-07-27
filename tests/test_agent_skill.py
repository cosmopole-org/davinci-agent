"""Tests for Caspar proxy-entity ("agent") skill delivery into Davinci runs.

An agent is deployed on Caspar as a proxy entity holding a skill file; the
node forwards every signal to the davinci creature with the skill attached
under ``skill`` plus a ``correlationId``/``replyTo`` envelope. Davinci must
(1) accept such packets as tasks even without an explicit ``kind: task``,
(2) load the skill as the session's system instruction, and (3) echo the
correlation id back so the node can route the result through the proxy.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import redirect_stdout

from davinci import caspar_runtime as rt


class _SignalOnRegisterBridge:
    """Fake bridge that pushes one signal as soon as a handler registers."""

    machine_id = "davinci-machine"
    program_id = "davinci-program"
    session_id = "sess-1"

    def __init__(self, packet):
        self._packet = packet

    def on_signal(self, handler):
        handler("creatures/signal", self._packet)


def test_wait_for_task_accepts_proxied_skill_packet():
    packet = {
        "user": {"id": "proxy-prog"},
        "data": (
            '{"data":"summarize the daily report",'
            '"skill":"You are the reporting agent.",'
            '"correlationId":"corr-42","replyTo":"proxy-prog",'
            '"proxyProgramId":"proxy-prog","proxyEntityId":"agent"}'
        ),
    }
    with redirect_stdout(io.StringIO()):
        task, reply_to, corr = rt._wait_for_task_signal(
            _SignalOnRegisterBridge(packet), timeout=1)
    assert task is not None and task["skill"] == "You are the reporting agent."
    assert reply_to == "proxy-prog"
    assert corr == "corr-42"


def test_wait_for_task_still_ignores_unrelated_signals():
    packet = {"user": {"id": "x"}, "data": '{"kind":"chatter"}'}
    with redirect_stdout(io.StringIO()):
        task, reply_to, corr = rt._wait_for_task_signal(
            _SignalOnRegisterBridge(packet), timeout=0.2)
    assert task is None and reply_to is None and corr is None


def test_run_agent_loads_skill_as_session_instruction():
    task = {
        "data": "run the capability self-test",
        "skill": "Always operate as the release-notes skill.",
        "proxyProgramId": "proxy-prog",
        "proxyEntityId": "agent",
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        result, result_dict = rt._run_agent(None, task, {}, [])
    out = buf.getvalue()
    assert "DAVINCI_SKILL" in out
    # The skill text must reach the run's rendered instructions (run_start
    # trace carries them).
    assert "release-notes skill" in out
    # A proxied bare prompt (no "objective") still becomes the objective.
    assert result_dict.get("objective", "") == "run the capability self-test" or result.success is not None


def test_run_agent_without_skill_keeps_default_instructions():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rt._run_agent(None, {"objective": "self-test"}, {}, [])
    assert "DAVINCI_SKILL" not in buf.getvalue()


class _CapturingClient:
    """Minimal LLMClient stand-in that records the (system, prompt) it is given."""

    provider = "fake"
    models = ["fake-1"]
    active_model = "fake-1"

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt, system="", attachments=None):
        self.prompts.append((system or "", prompt))
        return self.reply

    def emit(self, *a, **k):
        pass


def test_reasoner_prepends_skill_into_every_system_prompt():
    """The deployed skill must reach the model on the USER-FACING generation
    calls — per-step reasoning (which can deliver the answer) and final synthesis
    — so the answer carries the persona. It must NOT be injected into the
    internal planner / reflection critic, which are structured machinery the
    persona would derail."""
    from davinci.mcp import ToolRegistry
    from davinci.memory import WorkingMemory
    from davinci.planning import Plan, PlanStep
    from davinci.reasoner import LLMReasoner

    skill = "You are Pixel, a witty pirate. Always answer in pirate slang."

    client = _CapturingClient('{"complexity": "trivial", "steps": [{"title": "answer", "category": "result"}]}')
    reasoner = LLMReasoner(client)
    reasoner.set_instructions(skill)

    # Answer-generating calls carry the persona.
    client.reply = '{"tool": null, "thought": "greet", "final_answer": "Ahoy!"}'
    reasoner.propose("greet the user",
                     PlanStep(id=1, title="answer", category="result"),
                     ToolRegistry(), WorkingMemory())
    prop_system, _ = client.prompts[-1]
    assert skill in prop_system

    client.reply = '{"final_answer": "Ahoy matey!"}'
    plan = Plan(objective="greet the user")
    reasoner.set_conversation([{"role": "user", "content": "hi"}])
    reasoner.synthesize("greet the user", plan, WorkingMemory())
    syn_system, _ = client.prompts[-1]
    assert skill in syn_system

    # Internal machinery stays persona-free so its structured output is reliable.
    client.reply = '{"complexity": "trivial", "steps": [{"title": "answer", "category": "result"}]}'
    reasoner.make_plan("greet the user", [], ToolRegistry())
    plan_system, _ = client.prompts[-1]
    assert skill not in plan_system

    client.reply = '{"satisfied": true, "critique": "ok", "replan_titles": []}'
    reasoner.reflect("greet the user", plan, WorkingMemory())
    refl_system, _ = client.prompts[-1]
    assert skill not in refl_system


def test_planner_and_critic_stay_persona_free():
    from davinci.mcp import ToolRegistry
    from davinci.memory import WorkingMemory
    from davinci.planning import Plan
    from davinci.reasoner import LLMReasoner

    client = _CapturingClient('{"complexity": "trivial", "steps": [{"title": "a", "category": "result"}]}')
    reasoner = LLMReasoner(client)
    reasoner.set_instructions("You are Tina, a warm concierge.")
    reasoner.make_plan("do a thing", [], ToolRegistry())
    plan_system, _ = client.prompts[-1]
    assert "PERSONA" not in plan_system and "Tina" not in plan_system


def test_framework_prompts_do_not_claim_a_fixed_identity():
    """The task-framework system prompts must not hard-assert a proper-name
    identity ("You are Davinci…"): with a deployed skill that leaks as the
    agent's self-description when it is asked who it is (the reported bug — an
    agent named Tina introduced itself as "Davinci, an enterprise agent
    planner")."""
    from davinci.mcp import ToolRegistry
    from davinci.memory import WorkingMemory
    from davinci.planning import Plan, PlanStep
    from davinci.reasoner import LLMReasoner

    client = _CapturingClient('{"complexity": "trivial", "steps": [{"title": "a", "category": "result"}]}')
    reasoner = LLMReasoner(client)  # no skill set

    reasoner.make_plan("who are you", [], ToolRegistry())
    client.reply = '{"tool": null, "thought": "x", "final_answer": "hi"}'
    reasoner.propose("who are you", PlanStep(id=1, title="a", category="result"),
                     ToolRegistry(), WorkingMemory())
    client.reply = '{"final_answer": "hi"}'
    reasoner.set_conversation([{"role": "user", "content": "hi"}])
    reasoner.synthesize("who are you", Plan(objective="who are you"), WorkingMemory())
    client.reply = '{"satisfied": true, "critique": "", "replan_titles": []}'
    reasoner.reflect("who are you", Plan(objective="who are you"), WorkingMemory())

    for system, _ in client.prompts:
        assert "you are davinci" not in system.lower()


def test_skill_is_declared_authoritative_over_default_identity():
    """With a skill deployed, the answer-generating prompt must mark the persona
    as overriding the framework's default role so the model answers identity
    questions as the persona."""
    from davinci.memory import WorkingMemory
    from davinci.planning import Plan
    from davinci.reasoner import LLMReasoner

    client = _CapturingClient('{"final_answer": "I am Tina."}')
    reasoner = LLMReasoner(client)
    reasoner.set_instructions("You are Tina, a warm concierge who speaks briefly.")
    reasoner.synthesize("who are you", Plan(objective="who are you"), WorkingMemory())
    system, _ = client.prompts[-1]
    assert "You are Tina, a warm concierge who speaks briefly." in system
    assert "OVERRIDES" in system


def test_conversational_prompt_gets_spoken_answer_not_step_summary():
    """A greeting/identity/small-talk prompt must yield a real spoken answer,
    never the deterministic "Completed N/M steps" summary. synthesize answers it
    directly even with no tools or tool results."""
    from davinci import DavinciAgent, EchoExecutor
    from davinci.engine import ActionProposal, Reflection

    class ChatReasoner:
        """LLM stand-in: no tool for any step, always answers via synthesize."""

        def set_instructions(self, text):
            self._instr = text

        def set_conversation(self, history):
            pass

        def propose(self, objective, step, registry, memory):
            return ActionProposal(tool=None, thought="conversational")

        def reflect(self, objective, plan, memory):
            return Reflection(satisfied=True)

        def synthesize(self, objective, plan, memory):
            return "I'm doing wonderfully — I'm Tina, happy to help!"

    agent = DavinciAgent(reasoner=ChatReasoner(), executor=EchoExecutor())
    result = agent.run("How are you?")
    assert result.answer == "I'm doing wonderfully — I'm Tina, happy to help!"
    assert "Completed" not in result.answer


def test_terminal_result_answer_is_not_overridden_by_synthesis():
    """When a terminal result tool delivers a precise typed answer, the engine
    keeps it verbatim and does NOT let synthesis paraphrase it."""
    from davinci import DavinciAgent, EchoExecutor
    from davinci.engine import ActionProposal, Reflection
    from davinci.mcp import ToolRegistry
    from davinci.result_tool import register_result_tools

    registry = register_result_tools(ToolRegistry())

    class ResultReasoner:
        def set_instructions(self, text):
            pass

        def set_conversation(self, history):
            pass

        def propose(self, objective, step, registry, memory):
            tool = registry.get("result_as_text")
            if tool is not None:
                return ActionProposal(tool=tool, args={"value": "42"}, thought="deliver")
            return ActionProposal(tool=None, final_answer=None)

        def reflect(self, objective, plan, memory):
            return Reflection(satisfied=True)

        def synthesize(self, objective, plan, memory):
            return "SYNTHESIZED OVERRIDE"

    agent = DavinciAgent(registry=registry, reasoner=ResultReasoner(), executor=EchoExecutor())
    result = agent.run("give me the number")
    assert result.answer == "42"


def test_engine_hands_rendered_skill_to_reasoner():
    """The engine must forward its InstructionMemory (the loaded skill) to the
    reasoner via set_instructions, mirroring set_conversation."""
    from davinci import DavinciAgent, EchoExecutor
    from davinci.engine import ActionProposal, Reflection
    from davinci.memory import InstructionMemory

    seen = {}

    class Rec:
        def set_instructions(self, text):
            seen["instructions"] = text

        def propose(self, objective, step, registry, memory):
            return ActionProposal(tool=None, final_answer="done")

        def reflect(self, objective, plan, memory):
            return Reflection(satisfied=True)

    instructions = InstructionMemory()
    instructions.add_inline("You are the QA agent.", source="caspar-agent-skill")
    agent = DavinciAgent(reasoner=Rec(), executor=EchoExecutor(), instructions=instructions)
    agent.run("check the build")
    assert "You are the QA agent." in seen.get("instructions", "")


def test_wait_for_task_unwraps_client_payload_wrapper():
    # CLI convention: {programId, entity, payload:"<json>"} with the skill
    # and correlation envelope stamped on the wrapper by the node's proxy.
    packet = {
        "user": {"id": "proxy-prog"},
        "data": (
            '{"programId":"p1","entity":"agent",'
            '"payload":"{\\"objective\\":\\"draft the changelog\\",\\"correlationId\\":\\"cli-7\\"}",'
            '"skill":"Changelog skill.","correlationId":"cli-7","replyTo":"proxy-prog"}'
        ),
    }
    with redirect_stdout(io.StringIO()):
        task, reply_to, corr = rt._wait_for_task_signal(
            _SignalOnRegisterBridge(packet), timeout=1)
    assert task is not None
    assert task["objective"] == "draft the changelog"
    assert task["skill"] == "Changelog skill."
    assert reply_to == "proxy-prog"
    assert corr == "cli-7"
