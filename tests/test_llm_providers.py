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
    OpenRouterClient,
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
    assert client_from_config({"openrouter_api_key": "k"}).provider == "openrouter"


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


def test_per_agent_llm_override_from_backend_task_config():
    """The exact ``config.llm`` block Nest attaches for a per-agent override.

    The backend emits ``{provider, api_key, models: [<model>]}`` (single-element
    ``models`` list from the admin's one "llm model" field). Davinci must select
    that provider/model/key so the agent runs on the admin-chosen LLM instead of
    the default env-var backbone.
    """
    cfg = {"llm": {"provider": "anthropic", "api_key": "sk-agent",
                   "models": ["claude-opus-4-8"]}}
    client = client_from_config(cfg)
    assert client is not None
    assert client.provider == "anthropic"
    assert client.api_key == "sk-agent"
    assert client.models == ["claude-opus-4-8"]


def test_agent_override_wins_over_env_default(monkeypatch):
    """When an agent carries an override, it beats the default env provider."""
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini")
    cfg = {"llm": {"provider": "openai", "api_key": "sk-agent",
                   "models": ["gpt-4o"]}}
    client = client_from_config(cfg)
    assert client.provider == "openai"
    assert client.api_key == "sk-agent"
    assert client.models == ["gpt-4o"]


def test_no_agent_override_falls_back_to_env_default(monkeypatch):
    """With no override in the task config, davinci uses the env-var default."""
    for var in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                "GROK_API_KEY", "XAI_API_KEY", "AGENTROUTER_API_KEY",
                "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic")
    client = client_from_config({})
    assert client is not None
    assert client.provider == "anthropic"
    assert client.api_key == "env-anthropic"


def test_openrouter_selection_from_provider_name_and_model_config():
    cfg = {"llm_provider": "openrouter", "openrouter_api_key": "k",
           "openrouter_models": "anthropic/claude-sonnet-4.6"}
    client = client_from_config(cfg)
    assert client.provider == "openrouter"
    assert client.models == ["anthropic/claude-sonnet-4.6"]


def test_make_client_aliases_and_unknown():
    assert make_client("google", "k").provider == "gemini"
    assert make_client("gpt", "k").provider == "openai"
    assert make_client("openrouter.ai", "k").provider == "openrouter"
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
    for cls, host in [(OpenAIClient, "api.openai.com"), (GrokClient, "api.x.ai"),
                      (OpenRouterClient, "openrouter.ai")]:
        url, body, headers = _build(cls("k"))
        assert host in url and url.endswith("/chat/completions")
        assert headers["Authorization"] == "Bearer k"
        assert body["messages"][0]["role"] == "system"
        assert body["response_format"] == {"type": "json_object"}
        text = cls("k")._extract_text({"choices": [{"message": {"content": "ok"}}]})
        assert text == "ok"


def test_openrouter_optional_attribution_headers():
    # Absent by default …
    _url, _body, headers = _build(OpenRouterClient("k"))
    assert "HTTP-Referer" not in headers and "X-Title" not in headers
    # … present when configured.
    client = OpenRouterClient("k", site_url="https://davinci.example",
                              site_name="Davinci")
    _url, _body, headers = _build(client)
    assert headers["HTTP-Referer"] == "https://davinci.example"
    assert headers["X-Title"] == "Davinci"


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


def test_make_plan_is_dynamic_trivial_request_gets_single_step():
    # For a trivial request the model returns a single result step; the planner
    # must NOT pad it with analysis/verification/synthesis scaffolding.
    client = _FakeClient([
        json.dumps({"complexity": "trivial", "steps": [
            {"title": "Reply 'Hello! How can I help?'", "category": "result",
             "rationale": "Direct conversational answer, no tools needed."}]}),
    ])
    reasoner = LLMReasoner(client)
    plan = reasoner.make_plan("say hi", [], ToolRegistry())
    assert len(plan.steps) == 1
    assert plan.steps[0].category == "result"


def test_make_plan_still_decomposes_complex_requests():
    # A complex request still yields a full multi-step plan ending in a result.
    client = _FakeClient([
        json.dumps({"complexity": "complex", "steps": [
            {"title": "Fetch the page", "category": "research", "rationale": "r"},
            {"title": "Compute the total", "category": "analysis", "rationale": "r"},
            {"title": "Deliver the number", "category": "result", "rationale": "r"}]}),
    ])
    reasoner = LLMReasoner(client)
    plan = reasoner.make_plan("multi-part task", [], ToolRegistry())
    assert len(plan.steps) == 3
    assert plan.steps[-1].category == "result"


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


def test_make_client_normalizes_invalid_openrouter_free_model():
    """`openrouter/free` is not a real OpenRouter model — every request 400s and
    silently drops davinci to the heuristic. Normalize it to the valid
    `openrouter/auto` so a common misconfig doesn't masquerade as tool failures."""
    from davinci.llm import make_client

    c = make_client("openrouter", "sk-test", models=["openrouter/free"])
    assert c.models == ["openrouter/auto"]
    # A real model id is left untouched.
    c2 = make_client("openrouter", "sk-test", models=["meta-llama/llama-3.1-8b-instruct:free"])
    assert c2.models == ["meta-llama/llama-3.1-8b-instruct:free"]
    # Normalization is openrouter-only.
    g = make_client("gemini", "sk-test", models=["openrouter/free"])
    assert g.models == ["openrouter/free"]
