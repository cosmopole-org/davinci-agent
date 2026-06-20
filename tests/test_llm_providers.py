"""Tests for the cross-LLM provider abstraction.

These verify that Davinci's reasoning is genuinely provider-agnostic: the same
:class:`LLMReasoner` drives every provider, each provider builds its own correct
request shape and parses its own response, provider selection works from config,
and a dead client degrades to the deterministic heuristic reasoner.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davinci.engine import ActionProposal
from davinci.llm import (  # noqa: E402
    AnthropicClient,
    GeminiClient,
    GrokClient,
    LLMClient,
    OpenAIClient,
    client_from_config,
    make_client,
)
from davinci.mcp import ToolRegistry  # noqa: E402
from davinci.memory import WorkingMemory  # noqa: E402
from davinci.planning import Plan, Planner  # noqa: E402
from davinci.reasoner import LLMReasoner, reasoner_from_config  # noqa: E402


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #

def test_auto_detects_each_provider_from_its_key():
    assert client_from_config({"gemini_api_key": "k"}).provider == "gemini"
    assert client_from_config({"anthropic_api_key": "k"}).provider == "anthropic"
    assert client_from_config({"openai_api_key": "k"}).provider == "openai"
    assert client_from_config({"grok_api_key": "k"}).provider == "grok"
    assert client_from_config({"xai_api_key": "k"}).provider == "grok"


def test_explicit_provider_name_wins_over_other_keys():
    cfg = {"llm_provider": "claude", "gemini_api_key": "g", "anthropic_api_key": "a"}
    assert client_from_config(cfg).provider == "anthropic"


def test_llm_block_with_inline_key_and_models():
    cfg = {"llm": {"provider": "xai", "api_key": "x", "models": "grok-foo,grok-bar"}}
    client = client_from_config(cfg)
    assert client.provider == "grok"
    assert client.models == ["grok-foo", "grok-bar"]


def test_no_key_yields_no_client_and_no_reasoner():
    assert client_from_config({}) is None
    assert reasoner_from_config({}) is None


def test_make_client_aliases_and_unknown():
    assert make_client("google", "k").provider == "gemini"
    assert make_client("gpt", "k").provider == "openai"
    try:
        make_client("llama", "k")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown provider should raise")


# --------------------------------------------------------------------------- #
# Per-provider request shape + response parsing
# --------------------------------------------------------------------------- #

def _build(client):
    url, payload, headers = client._build_request(
        client.models[0], "SYSTEM", "PROMPT", [], True)
    return url, json.loads(payload), headers


def test_gemini_request_and_response_shape():
    url, body, headers = _build(GeminiClient("k"))
    assert "generativelanguage.googleapis.com" in url and "key=k" in url
    assert body["contents"][0]["parts"][-1]["text"] == "PROMPT"
    assert body["systemInstruction"]["parts"][0]["text"] == "SYSTEM"
    text = GeminiClient("k")._extract_text(
        {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
    assert text == "hi"


def test_anthropic_request_omits_sampling_and_uses_headers():
    url, body, headers = _build(AnthropicClient("k"))
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "k"
    assert headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "SYSTEM"
    assert body["model"] == "claude-opus-4-8"
    assert "max_tokens" in body
    # Sampling params are rejected (400) on current Claude models — must be absent.
    assert "temperature" not in body and "top_p" not in body
    text = AnthropicClient("k")._extract_text(
        {"content": [{"type": "text", "text": "claude"}]})
    assert text == "claude"


def test_openai_compatible_request_and_response():
    for cls, host in [(OpenAIClient, "api.openai.com"), (GrokClient, "api.x.ai")]:
        url, body, headers = _build(cls("k"))
        assert host in url and url.endswith("/chat/completions")
        assert headers["Authorization"] == "Bearer k"
        assert body["messages"][0]["role"] == "system"
        assert body["response_format"] == {"type": "json_object"}
        text = cls("k")._extract_text({"choices": [{"message": {"content": "ok"}}]})
        assert text == "ok"


# --------------------------------------------------------------------------- #
# Shared reasoner logic over a fake (provider-free) client
# --------------------------------------------------------------------------- #

class _FakeClient(LLMClient):
    """A scripted client: returns canned replies, no network."""

    provider = "fake"
    DEFAULT_MODELS = ["fake-1"]

    def __init__(self, replies):
        # bypass api_key requirement of the base ctor
        self.api_key = "x"
        self.models = ["fake-1"]
        self.temperature = 0.0
        self.timeout = 1
        self.max_retries = 1
        self.max_tokens = 16
        self.active_model = "fake-1"
        self._replies = list(replies)
        self.prompts = []

    def generate(self, prompt, *, system="", attachments=None, response_json=True):
        self.prompts.append(prompt)
        return self._replies.pop(0) if self._replies else None


def test_reasoner_make_plan_and_propose_are_provider_independent():
    client = _FakeClient([
        json.dumps({"steps": [
            {"title": "Fetch X", "category": "research", "rationale": "r"},
            {"title": "Answer", "category": "synthesis", "rationale": "r"}]}),
        json.dumps({"tool": None, "args": {}, "thought": "thinking",
                    "final_answer": None}),
    ])
    reasoner = LLMReasoner(client)
    assert reasoner.provider == "fake"

    plan = reasoner.make_plan("do it", ["research"], ToolRegistry())
    assert isinstance(plan, Plan)
    assert plan.steps[-1].category == "synthesis"

    prop = reasoner.propose("do it", plan.steps[0], ToolRegistry(), WorkingMemory())
    assert isinstance(prop, ActionProposal)
    assert prop.thought == "thinking"


def test_reasoner_falls_back_to_heuristic_when_client_dead():
    reasoner = LLMReasoner(_FakeClient([]))  # generate() always returns None
    plan = reasoner.make_plan("obj", ["research"], ToolRegistry())
    # heuristic plan still produced
    assert isinstance(plan, Plan) and plan.steps
    # propose on a non-synthesis step falls back without raising
    step = Planner._heuristic_plan("obj", ["research"]).steps[0]
    prop = reasoner.propose("obj", step, ToolRegistry(), WorkingMemory())
    assert isinstance(prop, ActionProposal)


def test_synthesize_uses_tool_results_and_obeys_format():
    client = _FakeClient([json.dumps({"final_answer": "42"})])
    reasoner = LLMReasoner(client)
    mem = WorkingMemory()
    mem.remember("tool_result", {"value": 42})
    plan = Plan(objective="o")
    plan.add_step("synth", "synthesis", "r")
    assert reasoner.synthesize("o", plan, mem) == "42"


# --------------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------------- #

def test_gemini_reasoner_back_compat_shim():
    from davinci.gemini_reasoner import GeminiReasoner, reasoner_from_config as rfc
    r = GeminiReasoner("fake-key")
    assert isinstance(r, LLMReasoner)
    assert r.provider == "gemini"
    assert r.models == GeminiClient.DEFAULT_MODELS
    assert rfc is reasoner_from_config
