"""Backward-compatible Gemini entry point.

Davinci's reasoning is now **cross-LLM**: the algorithms live once in
:class:`davinci.reasoner.LLMReasoner` and run on any
:class:`davinci.llm.LLMClient` (Gemini, Anthropic/Claude, OpenAI, Grok, …).

This module is kept as a thin shim so existing imports
(``from davinci.gemini_reasoner import GeminiReasoner, reasoner_from_config``)
keep working. ``GeminiReasoner`` is just an :class:`LLMReasoner` wired to a
:class:`~davinci.llm.GeminiClient`, with a couple of legacy proxy methods.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .llm import GeminiClient
from .llm import GeminiClient as _GeminiClient  # noqa: F401 — re-export
from .reasoner import LLMReasoner, reasoner_from_config  # noqa: F401 — re-export

# Preserved for callers that imported these constants from here.
DEFAULT_MODELS = GeminiClient.DEFAULT_MODELS
API_ROOT = GeminiClient.API_ROOT


class GeminiReasoner(LLMReasoner):
    """An :class:`LLMReasoner` backed by Gemini (kept for backward compatibility).

    New code should prefer :func:`davinci.reasoner.reasoner_from_config` (which
    picks the provider from config) or construct ``LLMReasoner(client)`` with the
    client of choice. This subclass keeps the historical constructor signature
    and the ``_generate`` / ``_attachment_part`` proxies some callers/tests use.
    """

    def __init__(self, api_key: str, *, models: Optional[List[str]] = None,
                 temperature: float = 0.2, timeout: int = 40,
                 max_retries: int = 3) -> None:
        super().__init__(GeminiClient(api_key, models=models, temperature=temperature,
                                      timeout=timeout, max_retries=max_retries))

    # -- legacy transport proxies (delegate to the Gemini client) ----------
    def _generate(self, prompt: str, *, system: str = "",
                  attachments: Optional[list] = None) -> Optional[str]:
        return self.client.generate(prompt, system=system, attachments=attachments)

    def _attachment_part(self, att: Any) -> Optional[dict]:
        return self.client._attachment_part(att)
