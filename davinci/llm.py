"""Cross-LLM client abstraction for Davinci.

This is the single transport-and-encoding seam that makes Davinci's reasoning,
planning and loop mechanisms **provider-agnostic**. The agent's algorithms live
once in :mod:`davinci.reasoner` (an :class:`~davinci.reasoner.LLMReasoner`); this
module supplies the per-provider plumbing they run on, so switching between
Gemini, Anthropic (Claude), OpenAI, Grok (xAI), or any OpenAI-compatible endpoint
is a one-line config change with **zero changes to the reasoning logic**.

Every provider implements the same tiny contract — :class:`LLMClient`:

    text = client.generate(prompt, system=..., attachments=..., response_json=True)

The base class owns everything that is identical across providers:

* **multi-model fallback + retry** — try each model in order, retry transient
  5xx with backoff, skip to the next model on quota/4xx;
* **stdlib-only transport** — ``urllib`` + ``ssl`` (the agent ships in a slim
  container that only adds ``pycryptodome``; no vendor SDKs);
* **proxy/CA awareness** — honours ``SSL_CERT_FILE`` / ``LLM_CA_BUNDLE`` so the
  egress-gateway CA baked into the image is trusted;
* **greppable, secret-free tracing** — one ``<PROVIDER> {json}`` sentinel line
  per LLM interaction for the supervising harness.

Each subclass supplies only what genuinely differs between providers:

* ``_build_request`` — endpoint URL, auth headers, request-body schema, and how
  attachments become that provider's multimodal "parts";
* ``_extract_text`` — pull the assistant text out of the provider's response.
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .attachments import Attachment, INLINE_MAX_BYTES


# --------------------------------------------------------------------------- #
# Shared transport helpers
# --------------------------------------------------------------------------- #

def _ssl_context() -> ssl.SSLContext:
    """Build a TLS context that trusts the egress-gateway CA when one is set."""
    cafile = (os.environ.get("LLM_CA_BUNDLE")
              or os.environ.get("GEMINI_CA_BUNDLE")
              or os.environ.get("SSL_CERT_FILE"))
    if cafile and os.path.isfile(cafile):
        return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def _emit(sentinel: str, event: str, **data: Any) -> None:
    """Greppable, secret-free trace line for the supervising harness.

    The sentinel is the provider name upper-cased (``GEMINI``, ``ANTHROPIC``,
    ``OPENAI``, ``GROK``) so existing tooling that greps ``GEMINI`` keeps working
    and new providers get the same treatment for free.
    """
    try:
        print(f"{sentinel} " + json.dumps({"event": event, **data}), flush=True)
    except Exception:  # pragma: no cover — never let tracing break the loop
        pass


# --------------------------------------------------------------------------- #
# Base client — provider-independent fallback + retry + transport
# --------------------------------------------------------------------------- #

class LLMClient:
    """Provider-agnostic LLM transport.

    Subclasses implement :meth:`_build_request` and :meth:`_extract_text`; the
    base class drives the same model-fallback + retry loop for everyone so the
    reasoning layer never has to know which provider it is talking to.
    """

    #: Provider identifier, e.g. ``"gemini"``; sets the trace sentinel.
    provider: str = "llm"
    #: Models tried in order; the first that answers wins.
    DEFAULT_MODELS: List[str] = []

    def __init__(self, api_key: str, *, models: Optional[List[str]] = None,
                 temperature: float = 0.2, timeout: int = 40,
                 max_retries: int = 3, max_tokens: int = 4096) -> None:
        if not api_key:
            raise ValueError(f"{type(self).__name__} requires a non-empty api_key")
        self.api_key = api_key
        self.models = list(models) if models else list(self.DEFAULT_MODELS)
        if not self.models:
            raise ValueError(f"{type(self).__name__} requires at least one model")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self._ctx = _ssl_context()
        self.active_model: Optional[str] = None

    @property
    def sentinel(self) -> str:
        return self.provider.upper()

    # -- public transport ---------------------------------------------------
    def generate(self, prompt: str, *, system: str = "",
                 attachments: Optional[List[Attachment]] = None,
                 response_json: bool = True) -> Optional[str]:
        """Send one prompt, trying each model in turn and retrying transient 5xx.

        Returns the assistant's text, or ``None`` if every model/attempt failed
        (the reasoning layer degrades to its deterministic heuristic on ``None``).
        """
        atts = list(attachments or [])
        for model in self.models:
            try:
                url, payload, headers = self._build_request(
                    model, system, prompt, atts, response_json)
            except Exception as exc:  # noqa: BLE001 — bad request shouldn't kill the loop
                _emit(self.sentinel, "build_error", model=model, error=repr(exc)[:160])
                continue
            for attempt in range(self.max_retries):
                try:
                    data = self._post(url, payload, headers)
                    text = self._extract_text(data)
                    if text:
                        self.active_model = model
                        return text
                    break  # 200 but empty/blocked — try next model
                except urllib.error.HTTPError as exc:
                    code = exc.code
                    # 5xx are transient → retry with backoff. Everything else
                    # (quota 429, auth 4xx) won't clear on retry → next model.
                    if code in (500, 502, 503, 529) and attempt < self.max_retries - 1:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    _emit(self.sentinel, "http_error", model=model, code=code)
                    break
                except Exception as exc:  # noqa: BLE001 — network/TLS/timeout
                    if attempt < self.max_retries - 1:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    _emit(self.sentinel, "transport_error", model=model,
                          error=repr(exc)[:160])
                    break
        return None

    def emit(self, event: str, **data: Any) -> None:
        """Emit a provider-tagged trace line (used by the reasoning layer)."""
        _emit(self.sentinel, event, **data)

    # -- transport primitive ------------------------------------------------
    def _post(self, url: str, payload: bytes, headers: Dict[str, str]) -> Dict[str, Any]:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as resp:
            return json.loads(resp.read().decode())

    @staticmethod
    def _b64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    @staticmethod
    def _attachment_bytes(att: Attachment) -> Optional[bytes]:
        try:
            return att.read_bytes()
        except OSError as exc:
            _emit("LLM", "attachment_read_error", name=att.name, error=repr(exc)[:160])
            return None

    # -- provider hooks (override) -----------------------------------------
    def _build_request(self, model: str, system: str, prompt: str,
                       attachments: List[Attachment],
                       response_json: bool) -> Tuple[str, bytes, Dict[str, str]]:
        raise NotImplementedError

    def _extract_text(self, data: Dict[str, Any]) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Gemini (Google)
# --------------------------------------------------------------------------- #

class GeminiClient(LLMClient):
    """Google Gemini via the ``generativelanguage`` REST API.

    Multimodal attachments are sent inline (``inlineData``, ≤ 18 MiB) or, for
    larger files, uploaded through the Gemini Files API and referenced by
    ``fileData.fileUri`` — the official multimodal contract.
    """

    provider = "gemini"
    DEFAULT_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-pro"]
    API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
    FILES_UPLOAD_ROOT = "https://generativelanguage.googleapis.com/upload/v1beta/files"

    def __init__(self, api_key: str, **kw: Any) -> None:
        super().__init__(api_key, **kw)
        # Cache of uploaded Files-API resources keyed by (path, size, mtime).
        self._file_cache: Dict[tuple, Dict[str, str]] = {}

    def _build_request(self, model, system, prompt, attachments, response_json):
        parts: List[Dict[str, Any]] = []
        for att in attachments:
            part = self._attachment_part(att)
            if part is not None:
                parts.append(part)
        parts.append({"text": prompt})
        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": self.temperature},
        }
        if response_json:
            body["generationConfig"]["responseMimeType"] = "application/json"
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{self.API_ROOT}/{model}:generateContent?key={self.api_key}"
        return url, json.dumps(body).encode(), {"Content-Type": "application/json"}

    def _extract_text(self, data: Dict[str, Any]) -> str:
        for cand in data.get("candidates", []) or []:
            for part in cand.get("content", {}).get("parts", []) or []:
                if isinstance(part.get("text"), str) and part["text"].strip():
                    return part["text"]
        return ""

    # -- multimodal: attachment → Gemini content part ----------------------
    def _attachment_part(self, att: Attachment) -> Optional[Dict[str, Any]]:
        data = self._attachment_bytes(att)
        if data is None:
            return None
        if len(data) <= INLINE_MAX_BYTES:
            return {"inlineData": {"mimeType": att.mime_type, "data": self._b64(data)}}
        cache_key = (att.path, len(data), self._mtime(att.path))
        ref = self._file_cache.get(cache_key)
        if ref is None:
            ref = self._upload_file(data, att)
            if ref is None:
                return None
            self._file_cache[cache_key] = ref
        return {"fileData": {"mimeType": ref.get("mimeType") or att.mime_type,
                             "fileUri": ref["fileUri"]}}

    @staticmethod
    def _mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    def _upload_file(self, data: bytes, att: Attachment) -> Optional[Dict[str, str]]:
        """Resumable upload to the Gemini Files API; returns ``{fileUri, mimeType}``."""
        meta = json.dumps({"file": {"displayName": att.name}}).encode()
        try:
            req = urllib.request.Request(
                f"{self.FILES_UPLOAD_ROOT}?key={self.api_key}", data=meta,
                headers={
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(len(data)),
                    "X-Goog-Upload-Header-Content-Type": att.mime_type,
                    "Content-Type": "application/json; charset=UTF-8",
                }, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as resp:
                upload_url = resp.headers.get("X-Goog-Upload-URL", "").strip()
            if not upload_url:
                self.emit("attachment_upload_error", name=att.name,
                          error="no upload URL in start response")
                return None
            req2 = urllib.request.Request(
                upload_url, data=data,
                headers={"Content-Length": str(len(data)), "X-Goog-Upload-Offset": "0",
                         "X-Goog-Upload-Command": "upload, finalize"}, method="POST")
            with urllib.request.urlopen(req2, timeout=self.timeout, context=self._ctx) as resp:
                body = json.loads(resp.read().decode())
            file_info = body.get("file") or body
            uri = file_info.get("uri") or ""
            if not uri:
                self.emit("attachment_upload_error", name=att.name,
                          error="no uri in finalize response")
                return None
            self.emit("attachment_uploaded", name=att.name, fileUri=uri, bytes=len(data))
            return {"fileUri": uri, "mimeType": file_info.get("mimeType") or att.mime_type}
        except Exception as exc:  # noqa: BLE001
            self.emit("attachment_upload_error", name=att.name, error=repr(exc)[:200])
            return None


# --------------------------------------------------------------------------- #
# Anthropic (Claude)
# --------------------------------------------------------------------------- #

class AnthropicClient(LLMClient):
    """Anthropic Claude via the Messages API (``POST /v1/messages``).

    Images are sent as ``image`` content blocks (base64 source), PDFs as
    ``document`` blocks; ``system`` is a top-level field. ``max_tokens`` is
    required. Sampling params are *not* sent — they are rejected (400) on the
    current Opus/Sonnet family, and omitting them is valid on every model.
    """

    provider = "anthropic"
    # Default to the most capable model, with a graceful cost/availability
    # fallback chain (Opus → Sonnet → Haiku).
    DEFAULT_MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def _build_request(self, model, system, prompt, attachments, response_json):
        content: List[Dict[str, Any]] = []
        for att in attachments:
            block = self._attachment_block(att)
            if block is not None:
                content.append(block)
        content.append({"type": "text", "text": prompt})
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            body["system"] = system
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
        }
        return self.API_URL, json.dumps(body).encode(), headers

    def _extract_text(self, data: Dict[str, Any]) -> str:
        for block in data.get("content", []) or []:
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                if block["text"].strip():
                    return block["text"]
        return ""

    def _attachment_block(self, att: Attachment) -> Optional[Dict[str, Any]]:
        data = self._attachment_bytes(att)
        if data is None:
            return None
        mime = att.mime_type or "application/octet-stream"
        source = {"type": "base64", "media_type": mime, "data": self._b64(data)}
        if mime == "application/pdf":
            return {"type": "document", "source": source}
        if mime.startswith("image/"):
            return {"type": "image", "source": source}
        # Other types aren't a native Claude block — skip rather than corrupt.
        self.emit("attachment_skipped", name=att.name, mime=mime)
        return None


# --------------------------------------------------------------------------- #
# OpenAI-compatible (OpenAI, Grok/xAI, and any chat/completions endpoint)
# --------------------------------------------------------------------------- #

class OpenAICompatibleClient(LLMClient):
    """Any OpenAI-style ``/chat/completions`` endpoint.

    Subclass and set :attr:`BASE_URL` + :attr:`DEFAULT_MODELS` to target a
    specific vendor (OpenAI, Grok/xAI, DeepSeek, Mistral, Together, Groq,
    Ollama, …) — the wire format is shared. Images are sent as ``image_url``
    parts with ``data:`` URIs.
    """

    provider = "openai"
    BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODELS = ["gpt-4o", "gpt-4o-mini"]

    def _build_request(self, model, system, prompt, attachments, response_json):
        user_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for att in attachments:
            part = self._attachment_part(att)
            if part is not None:
                user_parts.append(part)
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_parts})
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_json:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        return f"{self.BASE_URL}/chat/completions", json.dumps(body).encode(), headers

    def _extract_text(self, data: Dict[str, Any]) -> str:
        for choice in data.get("choices", []) or []:
            content = (choice.get("message") or {}).get("content")
            if isinstance(content, str) and content.strip():
                return content
            # Some vendors return a parts array for multimodal replies.
            if isinstance(content, list):
                for part in content:
                    if isinstance(part.get("text"), str) and part["text"].strip():
                        return part["text"]
        return ""

    def _attachment_part(self, att: Attachment) -> Optional[Dict[str, Any]]:
        data = self._attachment_bytes(att)
        if data is None:
            return None
        mime = att.mime_type or "application/octet-stream"
        if not mime.startswith("image/"):
            # chat/completions only accepts images inline; skip other types.
            self.emit("attachment_skipped", name=att.name, mime=mime)
            return None
        uri = f"data:{mime};base64,{self._b64(data)}"
        return {"type": "image_url", "image_url": {"url": uri}}


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI's hosted ``api.openai.com`` endpoint."""

    provider = "openai"
    BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODELS = ["gpt-4o", "gpt-4o-mini"]


class GrokClient(OpenAICompatibleClient):
    """xAI Grok — OpenAI-compatible API at ``api.x.ai``."""

    provider = "grok"
    BASE_URL = "https://api.x.ai/v1"
    DEFAULT_MODELS = ["grok-2-latest", "grok-2-vision-latest"]


# --------------------------------------------------------------------------- #
# Provider registry + construction
# --------------------------------------------------------------------------- #

#: provider name → (client class, config/env key aliases for its API key).
PROVIDERS: Dict[str, Tuple[type, List[str], List[str]]] = {
    # name:        (class,            api_key config keys,        api_key env vars)
    "gemini":    (GeminiClient,    ["gemini_api_key"],         ["GEMINI_API_KEY"]),
    "anthropic": (AnthropicClient, ["anthropic_api_key"],      ["ANTHROPIC_API_KEY"]),
    "openai":    (OpenAIClient,    ["openai_api_key"],         ["OPENAI_API_KEY"]),
    "grok":      (GrokClient,      ["grok_api_key", "xai_api_key"],
                                                               ["GROK_API_KEY", "XAI_API_KEY"]),
}

#: friendly aliases users might pass for ``llm_provider``.
PROVIDER_ALIASES = {
    "google": "gemini", "gemini": "gemini",
    "claude": "anthropic", "anthropic": "anthropic",
    "openai": "openai", "gpt": "openai", "chatgpt": "openai",
    "grok": "grok", "xai": "grok", "x.ai": "grok",
}


def _api_key_for(provider: str, config: Dict[str, Any]) -> str:
    """Resolve a provider's API key from config (flat or nested) or environment."""
    _cls, cfg_keys, env_keys = PROVIDERS[provider]
    nested = config.get(provider) if isinstance(config.get(provider), dict) else {}
    for key in cfg_keys:
        val = config.get(key) or nested.get("api_key")
        if val:
            return str(val).strip()
    if nested.get("api_key"):
        return str(nested["api_key"]).strip()
    for env in env_keys:
        if os.environ.get(env):
            return os.environ[env].strip()
    return ""


def _models_for(provider: str, config: Dict[str, Any]) -> Optional[List[str]]:
    """Resolve an optional model fallback list for a provider from config/env."""
    nested = config.get(provider) if isinstance(config.get(provider), dict) else {}
    raw = (config.get(f"{provider}_models") or nested.get("models")
           or config.get("llm_models"))
    if isinstance(raw, str):
        raw = [m.strip() for m in raw.split(",") if m.strip()]
    if raw:
        return list(raw)
    env_one = os.environ.get(f"{provider.upper()}_MODEL")
    return [env_one] if env_one else None


def make_client(provider: str, api_key: str, *, models: Optional[List[str]] = None,
                **kw: Any) -> LLMClient:
    """Construct the client for a named provider (raises on unknown provider)."""
    canonical = PROVIDER_ALIASES.get(provider.lower().strip())
    if canonical is None:
        raise ValueError(f"unknown LLM provider: {provider!r} "
                         f"(known: {sorted(set(PROVIDER_ALIASES.values()))})")
    cls = PROVIDERS[canonical][0]
    return cls(api_key, models=models, **kw)


def client_from_config(config: Dict[str, Any]) -> Optional[LLMClient]:
    """Build the right :class:`LLMClient` from a task config / environment.

    Selection order:

    1. an explicit ``llm`` block ``{provider, api_key, models}``;
    2. an explicit ``llm_provider`` / ``provider`` name (key taken from the
       matching ``<provider>_api_key`` / ``<PROVIDER>_API_KEY``);
    3. otherwise the first provider that has a usable key (Gemini, Anthropic,
       OpenAI, Grok), so existing Gemini-only configs keep working unchanged.

    Returns ``None`` when no provider has a key — the agent then runs on its
    deterministic heuristic reasoner.
    """
    config = config or {}

    # 1. generic { "llm": { provider, api_key, models } } block.
    llm = config.get("llm") if isinstance(config.get("llm"), dict) else None
    if llm and llm.get("provider"):
        provider = PROVIDER_ALIASES.get(str(llm["provider"]).lower().strip())
        if provider:
            api_key = (str(llm.get("api_key") or "").strip()
                       or _api_key_for(provider, config))
            if api_key:
                models = llm.get("models") or _models_for(provider, config)
                if isinstance(models, str):
                    models = [m.strip() for m in models.split(",") if m.strip()]
                return make_client(provider, api_key, models=models)

    # 2. explicit provider name elsewhere in config / env.
    named = (config.get("llm_provider") or config.get("provider")
             or os.environ.get("LLM_PROVIDER"))
    if named:
        provider = PROVIDER_ALIASES.get(str(named).lower().strip())
        if provider:
            api_key = _api_key_for(provider, config)
            if api_key:
                return make_client(provider, api_key, models=_models_for(provider, config))

    # 3. auto-detect: first provider that has a key. Gemini first for
    #    backward compatibility with existing deployments.
    for provider in ("gemini", "anthropic", "openai", "grok"):
        api_key = _api_key_for(provider, config)
        if api_key:
            return make_client(provider, api_key, models=_models_for(provider, config))
    return None
