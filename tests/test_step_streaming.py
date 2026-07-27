"""Tests for per-step streaming of a Davinci run.

A run is a multi-step loop; every trajectory event (the model's thought, the
tool it chose, the tool's result, the final answer) must be streamed back to
the requester over the gateway AS IT HAPPENS, so a client never has to hold a
socket open for the whole run. Streaming rides the tracer's sink: each
``TraceEvent`` is signalled to the user's creature as a ``davinci/step`` packet
carrying the run's correlation id.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import redirect_stdout

from davinci import caspar_runtime as rt
from davinci.observability import Tracer


class _RecordingBridge:
    """Fake bridge that records every ``signal_user`` call."""

    machine_id = "davinci-machine"
    program_id = "davinci-program"
    session_id = "sess-1"

    def __init__(self):
        self.sent = []

    def signal_user(self, key, user_id, packet):
        self.sent.append((key, user_id, packet))
        return {"ok": True}


def test_channel_mapping_groups_kinds():
    assert rt._stream_channel("reason") == "thought"
    assert rt._stream_channel("reflection") == "thought"
    assert rt._stream_channel("tool_result") == "observation"
    assert rt._stream_channel("run_end") == "final"
    assert rt._stream_channel("decision") == "action"
    assert rt._stream_channel("something_new") == "trace"


def test_sink_streams_each_event_to_the_user():
    bridge = _RecordingBridge()
    tracer = Tracer(sink=rt._make_step_sink(bridge, "user-99", "corr-7"))
    tracer.emit("reason", "thinking about the task")
    tracer.emit("tool_result", "web_search ok", ok=True)

    assert len(bridge.sent) == 2
    key, user_id, packet = bridge.sent[0]
    assert key == "creatures/signal"
    assert user_id == "user-99"
    assert packet["kind"] == "davinci/step"
    # Non-terminal multi-response chunk so the proxy keeps the correlation open.
    assert packet["stream"] is True and packet["final"] is False
    assert packet["correlationId"] == "corr-7"
    assert packet["channel"] == "thought"
    assert packet["seq"] == 1
    assert packet["event"]["message"] == "thinking about the task"
    # Final observation carries the tool outcome the user should see.
    assert bridge.sent[1][2]["channel"] == "observation"
    assert bridge.sent[1][2]["seq"] == 2


def test_streamto_disabled_or_missing_yields_no_sink():
    # No target -> no streaming.
    assert rt._make_step_sink(_RecordingBridge(), "", "corr") is None
    # No bridge -> no streaming.
    assert rt._make_step_sink(None, "user", "corr") is None


def test_stream_steps_env_flag_off(monkeypatch=None):
    os.environ["DAVINCI_STREAM_STEPS"] = "0"
    try:
        assert rt._make_step_sink(_RecordingBridge(), "user", "corr") is None
    finally:
        os.environ.pop("DAVINCI_STREAM_STEPS", None)


def test_sink_swallows_bridge_failures():
    class _BoomBridge(_RecordingBridge):
        def signal_user(self, key, user_id, packet):
            raise RuntimeError("gateway down")

    sink = rt._make_step_sink(_BoomBridge(), "user", "corr")
    # A tracer emit through a failing sink must not raise.
    with redirect_stdout(io.StringIO()):
        Tracer(sink=sink).emit("reason", "still fine")
