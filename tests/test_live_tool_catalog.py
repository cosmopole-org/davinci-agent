"""A live agent's tool catalog is exactly what the orchestrator sent it.

The failure this guards against is not a crash: an agent that falls back to the
built-in demo capabilities tells its users, confidently, that it can search the
web and run Python. There is no creature behind those names, so the claim is
false and the call would fail. On a live run an empty catalog must mean "no
tools", and a catalog with the space's sandbox in it must mean exactly that one.
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davinci import caspar_runtime as rt


class _FakeBridge:
    """A gateway that answers tool invocations the way a live tool creature does.

    Replying matters: the executor registers its waiter *before* signalling and
    then blocks for `max_exec_seconds`, so a silent fake would just stall.
    """

    machine_id = "davinci-machine"
    program_id = "davinci-program"
    session_id = "sess-1"

    def __init__(self):
        self.sent = []
        self._handler = None

    def on_signal(self, handler):
        self._handler = handler

    def signal_user(self, key, user_id, packet):
        self.sent.append({"key": key, "to": user_id, "packet": packet})
        if isinstance(packet, dict) and packet.get("kind") == "invoke" and self._handler:
            self._handler("creatures/signal", {
                "kind": "tools/result",
                "correlationId": packet.get("correlationId"),
                "result": {"tool_id": packet.get("tool_id"), "function": packet.get("function"),
                           "result": {"ok": True, "stdout": "hello\n", "exit_code": 0}},
            })
        return {"ok": True}

    def invocations(self):
        return [s["packet"] for s in self.sent if s["packet"].get("kind") == "invoke"]


SANDBOX_ENTRY = {
    "name": "sandbox",
    "tool_id": "212@global",
    "kind": "tool",
    "category": "execution",
    "description": "Use when: you need to actually run something. How to talk to it: "
                   "signal it with `function` naming the operation.",
    "arg_schema": {"function": {"type": "string", "description": "exec | write | read"},
                   "command": {"type": "string", "description": "the shell command line"}},
    "required": ["function"],
    "risk": "high",
    "requires_network": True,
    "program_id": "212@global",
    "creature_id": "210@global",
    "entity_id": "vercel_sandbox",
    "function": "exec",
    "defaults": {"space_id": "42@global"},
}


def _run(config, objective="run the test suite"):
    """Run the agent live against a fake bridge; return (snapshot, bridge)."""
    bridge = _FakeBridge()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rt._run_agent(bridge, {"objective": objective}, config, [])
    for line in buf.getvalue().splitlines():
        if line.startswith("DAVINCI_BOOT") and "capabilities" in line:
            return json.loads(line.split("DAVINCI_BOOT", 1)[1])["capabilities"], bridge
    raise AssertionError("no DAVINCI_BOOT capability snapshot emitted")


def test_empty_catalog_means_no_tools_not_demo_tools():
    snapshot, bridge = _run({"tools": []})

    assert snapshot["tools"] == [] and snapshot["tool_count"] == 0
    # The demo placeholders must not appear anywhere in what the agent believes
    # it can do — this is what made an agent claim web search and python exec.
    blob = repr(snapshot)
    for phantom in ("web_search", "python_exec", "git_tool", "vector_search"):
        assert phantom not in blob, f"{phantom} leaked into a live run: {snapshot}"
    assert bridge.invocations() == []


def test_the_space_sandbox_is_the_catalog_when_it_is_the_only_tool():
    snapshot, _ = _run({"tools": [SANDBOX_ENTRY]})
    assert snapshot["tools"] == ["sandbox"] and snapshot["tool_count"] == 1
    assert "web_search" not in repr(snapshot)


def test_invoking_the_sandbox_targets_its_program_and_pins_the_space():
    _, bridge = _run({"tools": [SANDBOX_ENTRY]})
    invokes = bridge.invocations()
    assert invokes, "the agent never signalled its only tool"

    packet = invokes[0]
    assert bridge.sent[0]["to"] == "212@global"        # the sandbox creature's program
    assert packet["entityId"] == "vercel_sandbox"      # so the node can cold-spawn it
    assert packet["function"] == "exec"                # the catalog's routing function
    # The space is pinned by the platform, never by the model.
    assert packet["payload"]["space_id"] == "42@global"


def test_sandbox_entry_becomes_a_usable_descriptor():
    """The reasoner picks a tool from its description and calls it with the
    schema's argument names, so both have to survive the catalog round-trip."""
    registry = rt._registry_from_tool_catalog([SANDBOX_ENTRY])
    tool = registry.get("sandbox")

    assert tool is not None
    assert "run something" in tool.description
    assert tool.category == "execution" and tool.risk == "high"
    schema = tool.schema()
    assert set(schema["properties"]) == {"function", "command"}
    # Only `function` is required: a status check must not have to invent a
    # command, and `write` must not have to invent one either.
    assert schema["required"] == ["function"]


def test_offline_runs_keep_the_demo_capabilities():
    """With no bridge there is no orchestrator to send a catalog, so the demo
    set is the point — it keeps `python3 -m davinci.caspar_runtime` useful."""
    names = [t.name for t in rt._build_registry().all()]
    assert any("web_search" in n for n in names)
    assert all(n.startswith("local__") for n in names), names
