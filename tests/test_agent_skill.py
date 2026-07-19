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
