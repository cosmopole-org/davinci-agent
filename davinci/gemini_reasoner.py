"""Gemini-backed reasoner — uses Google's Gemini API as the agent's LLM backbone.

This is the "real LLM backend" plug-point referenced in the engine docs: it
implements the :class:`~davinci.engine.Reasoner` protocol (``propose`` +
``reflect``) by asking a Gemini model to choose the right tool (and arguments)
for each plan step and to self-critique the finished plan.

Design constraints:

* **stdlib-only** — the agent ships in a slim container that only adds
  ``pycryptodome``; this module therefore talks to the Gemini REST API with
  ``urllib`` + ``ssl`` rather than the ``google-generativeai`` SDK.
* **proxy/CA aware** — outbound HTTPS in hardened environments is intercepted by
  an egress gateway, so the TLS context honours ``SSL_CERT_FILE`` /
  ``GEMINI_CA_BUNDLE`` (a CA bundle baked into the image).
* **degrade gracefully** — any network/quota/parse failure falls back to the
  deterministic :class:`~davinci.engine.HeuristicReasoner`, so a flaky LLM never
  breaks the agent loop. Every LLM interaction is surfaced on stdout as a
  ``GEMINI`` sentinel line for the supervising harness.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .attachments import Attachment, INLINE_MAX_BYTES
from .engine import ActionProposal, HeuristicReasoner, Reflection
from .mcp import ToolRegistry
from .memory import WorkingMemory
from .planning import Plan, PlanStep, StepStatus

# Models tried in order — ``gemini-flash-latest`` is the stable alias that stays
# available when a pinned snapshot is rate-limited or temporarily unavailable.
DEFAULT_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-pro"]
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
FILES_UPLOAD_ROOT = "https://generativelanguage.googleapis.com/upload/v1beta/files"
FILES_GET_ROOT = "https://generativelanguage.googleapis.com/v1beta"


def _ssl_context() -> ssl.SSLContext:
    """Build a TLS context that trusts the egress-gateway CA when one is set."""
    cafile = os.environ.get("GEMINI_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if cafile and os.path.isfile(cafile):
        return ssl.create_default_context(cafile=cafile)
    # ``create_default_context`` already honours SSL_CERT_FILE/SSL_CERT_DIR via
    # the OpenSSL default paths; this is the generic fallback.
    return ssl.create_default_context()


class GeminiReasoner:
    """A :class:`Reasoner` that drives tool selection with a Gemini model."""

    def __init__(self, api_key: str, *, models: Optional[List[str]] = None,
                 temperature: float = 0.2, timeout: int = 40,
                 max_retries: int = 3) -> None:
        if not api_key:
            raise ValueError("GeminiReasoner requires a non-empty api_key")
        self.api_key = api_key
        self.models = models or list(DEFAULT_MODELS)
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._ctx = _ssl_context()
        self._fallback = HeuristicReasoner()
        self.active_model: Optional[str] = None
        # Cache of uploaded Gemini Files API resources keyed by (path, size, mtime)
        # so repeat propose/reflect calls don't re-upload the same attachment.
        self._file_cache: Dict[tuple, Dict[str, str]] = {}

    # -- LLM transport -------------------------------------------------------
    def _generate(self, prompt: str, *, system: str = "",
                  attachments: Optional[List[Attachment]] = None) -> Optional[str]:
        """POST one prompt to Gemini, trying each model + retrying transient 5xx.

        ``attachments`` are the multimodal parts that accompany the user
        message; each is sent inline (<= 18 MiB) or, for larger files, uploaded
        via the Gemini Files API and referenced as ``fileData`` — the standard
        contract documented at
        https://ai.google.dev/gemini-api/docs/file-prompting-strategies.
        """
        parts: List[Dict[str, Any]] = []
        for att in (attachments or []):
            part = self._attachment_part(att)
            if part is not None:
                parts.append(part)
        parts.append({"text": prompt})
        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": self.temperature,
                                  "responseMimeType": "application/json"},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        payload = json.dumps(body).encode()

        for model in self.models:
            url = f"{API_ROOT}/{model}:generateContent?key={self.api_key}"
            for attempt in range(self.max_retries):
                try:
                    req = urllib.request.Request(
                        url, data=payload,
                        headers={"Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as resp:
                        data = json.loads(resp.read().decode())
                    text = self._extract_text(data)
                    if text:
                        self.active_model = model
                        return text
                    break  # 200 but empty/blocked — try next model
                except urllib.error.HTTPError as exc:
                    code = exc.code
                    # 429 = quota/rate limit: retrying the *same* model rarely
                    # clears within the loop, so fall straight through to the
                    # next model. 500/503 are transient — retry with backoff.
                    if code in (500, 503) and attempt < self.max_retries - 1:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    _emit("http_error", model=model, code=code)
                    break  # non-retryable / quota / exhausted — next model
                except Exception as exc:  # noqa: BLE001 — network/TLS/timeout
                    if attempt < self.max_retries - 1:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    _emit("transport_error", model=model, error=repr(exc)[:160])
                    break
        return None

    # -- multimodal: attachment → Gemini content part ----------------------
    def _attachment_part(self, att: Attachment) -> Optional[Dict[str, Any]]:
        """Return a Gemini ``parts[]`` entry for an attachment, or ``None``.

        Files <= ``INLINE_MAX_BYTES`` are inlined as ``inlineData``; larger
        files go through the Files API (resumable upload) and are referenced by
        ``fileData.fileUri``. Either way the model receives a real multimodal
        part, the same as the official SDKs would produce.
        """
        try:
            data = att.read_bytes()
        except OSError as exc:
            _emit("attachment_read_error", name=att.name, error=repr(exc)[:160])
            return None

        if len(data) <= INLINE_MAX_BYTES:
            import base64 as _b64
            return {"inlineData": {"mimeType": att.mime_type,
                                    "data": _b64.b64encode(data).decode("ascii")}}

        # Large file: upload to the Files API and reference by fileUri.
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
                f"{FILES_UPLOAD_ROOT}?key={self.api_key}",
                data=meta,
                headers={
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(len(data)),
                    "X-Goog-Upload-Header-Content-Type": att.mime_type,
                    "Content-Type": "application/json; charset=UTF-8",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout,
                                         context=self._ctx) as resp:
                upload_url = resp.headers.get("X-Goog-Upload-URL", "").strip()
            if not upload_url:
                _emit("attachment_upload_error", name=att.name,
                      error="no upload URL in start response")
                return None
            req2 = urllib.request.Request(
                upload_url, data=data,
                headers={
                    "Content-Length": str(len(data)),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                },
                method="POST",
            )
            with urllib.request.urlopen(req2, timeout=self.timeout,
                                         context=self._ctx) as resp:
                body = json.loads(resp.read().decode())
            file_info = body.get("file") or body
            uri = file_info.get("uri") or ""
            mime = file_info.get("mimeType") or att.mime_type
            if not uri:
                _emit("attachment_upload_error", name=att.name,
                      error="no uri in finalize response")
                return None
            _emit("attachment_uploaded", name=att.name,
                  fileUri=uri, bytes=len(data))
            return {"fileUri": uri, "mimeType": mime}
        except Exception as exc:  # noqa: BLE001
            _emit("attachment_upload_error", name=att.name,
                  error=repr(exc)[:200])
            return None

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        for cand in data.get("candidates", []) or []:
            for part in cand.get("content", {}).get("parts", []) or []:
                if isinstance(part.get("text"), str) and part["text"].strip():
                    return part["text"]
        return ""

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            # Last resort: grab the outermost {...} span.
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(text[start:end + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
            return None

    # -- Reasoner protocol ---------------------------------------------------
    def propose(self, objective: str, step: PlanStep, registry: ToolRegistry,
                memory: WorkingMemory) -> ActionProposal:
        # The final synthesis is handled once, at end-of-run, with the full
        # untruncated tool results (see ``synthesize``). Skipping a per-step
        # answer here avoids committing to a low-quality answer built from the
        # truncated step memory.
        if step.category == "synthesis":
            return ActionProposal(
                tool=None,
                thought="final answer deferred to end-of-run synthesis over full results")
        # Include each tool's input schema so the model emits arguments that match
        # the tool's real contract (this is what makes Path A / schema-rich
        # discovery pay off — e.g. python_exec wants `code`, web_search `query`).
        catalog = []
        for t in registry.all():
            entry = t.stub()
            try:
                schema = t.schema()
            except Exception:  # noqa: BLE001
                schema = {}
            props = (schema or {}).get("properties") if isinstance(schema, dict) else None
            if props:
                entry["arg_schema"] = props
                entry["required_args"] = (schema or {}).get("required", [])
            catalog.append(entry)
        attachments = self._attachments_from_memory(memory)
        system = (
            "You are Davinci, an enterprise agent planner. For the CURRENT step you "
            "pick exactly one registered tool to advance the objective, or no tool for "
            "pure-cognition (analysis/synthesis/verification) steps. Always reply with "
            "a single JSON object: {\"tool\": <tool name or null>, \"args\": {object}, "
            "\"thought\": <short reason>, \"final_answer\": <string or null>}. "
            "Choose the least-risk tool that fits the step's category. CRITICAL: fill "
            "\"args\" using the chosen tool's \"arg_schema\" (use those exact property "
            "names and types) — e.g. provide runnable code for a code arg, a real query "
            "string for a query arg. Only fall back to a generic args.task when the tool "
            "exposes no arg_schema. When the user supplied attachments, they are sent "
            "alongside this message as standard multimodal parts — inspect them when the "
            "step needs their content."
        )
        prompt_payload: Dict[str, Any] = {
            "objective": objective,
            "current_step": {"title": step.title, "category": step.category,
                             "rationale": step.rationale},
            "available_tools": catalog,
            "recent_memory": self._memory_digest(memory),
        }
        if attachments:
            prompt_payload["attachments"] = [
                {"name": a.name, "mime_type": a.mime_type, "size": a.size,
                 "kind": a.kind, "description": a.description}
                for a in attachments
            ]
        prompt = json.dumps(prompt_payload, indent=2)

        text = self._generate(prompt, system=system, attachments=attachments)
        decision = self._parse_json(text) if text else None
        if not decision:
            _emit("propose_fallback", step=step.title)
            return self._fallback.propose(objective, step, registry, memory)

        thought = str(decision.get("thought") or f"[gemini:{self.active_model}] {step.title}")
        final_answer = decision.get("final_answer")
        if isinstance(final_answer, str) and final_answer.strip():
            _emit("propose", model=self.active_model, step=step.title, decision="final_answer")
            return ActionProposal(tool=None, thought=thought, final_answer=final_answer.strip())

        tool_name = decision.get("tool")
        tool = registry.get(tool_name) if isinstance(tool_name, str) else None
        if tool_name and not tool:
            # Model named a tool that isn't in the registry — recover by search.
            matches = registry.search(str(tool_name), max_results=1)
            tool = matches[0] if matches else None
        args = decision.get("args") if isinstance(decision.get("args"), dict) else {}
        args.setdefault("task", objective)
        args.setdefault("step", step.title)
        _emit("propose", model=self.active_model, step=step.title,
              tool=tool.name if tool else None)
        return ActionProposal(tool=tool, args=args, thought=thought)

    def make_plan(self, objective: str, required_categories: List[str],
                  registry: ToolRegistry) -> Plan:
        """LLM-backed planning strategy: decompose the *actual* objective into a
        concrete, ordered, executable sequence of steps.

        This is what lets the agent do real work: instead of generic
        ``Execute '<category>' work`` placeholders, each step carries a concrete
        instruction (which URL to fetch, what to extract, what to compute), so
        the per-step ``propose`` call can emit real tool arguments. Falls back to
        the deterministic heuristic plan on any LLM/parse failure.
        """
        from .planning import Planner as _Planner  # local import avoids a cycle

        tools = [t.stub() for t in registry.all()]
        categories = sorted({t.get("category", "general") for t in tools} |
                            {"analysis", "verification", "synthesis"})
        system = (
            "You are Davinci's planner. Decompose the objective into a CONCRETE, "
            "ordered, minimal sequence of executable steps that actually accomplish "
            "it end to end using the available tools. Each step's \"title\" must be a "
            "specific, actionable instruction (name the exact URL, data, or "
            "computation — never a vague 'do the research' placeholder). Pick each "
            "step's \"category\" from the provided list so it routes to a matching "
            "tool. Keep it tight (5-9 steps). The LAST step MUST have category "
            "\"synthesis\" and produce the final answer, honoring any output-format "
            "constraint stated in the objective. Reply with a single JSON object: "
            "{\"steps\": [{\"title\": str, \"category\": str, \"rationale\": str}, ...]}."
        )
        prompt = json.dumps({
            "objective": objective,
            "available_categories": categories,
            "available_tools": tools,
        }, indent=2)
        text = self._generate(prompt, system=system)
        decision = self._parse_json(text) if text else None
        steps = (decision or {}).get("steps") if isinstance(decision, dict) else None
        if not isinstance(steps, list) or not steps:
            _emit("plan_fallback")
            return _Planner._heuristic_plan(objective, required_categories)

        plan = Plan(objective=objective)
        added = 0
        for s in steps:
            if not isinstance(s, dict):
                continue
            title = str(s.get("title") or "").strip()
            if not title:
                continue
            plan.add_step(title, str(s.get("category") or "general").strip() or "general",
                          str(s.get("rationale") or "")[:300])
            added += 1
        if added == 0:
            return _Planner._heuristic_plan(objective, required_categories)
        if plan.steps[-1].category != "synthesis":
            plan.add_step("Synthesize the final answer in the exact required format",
                          "synthesis", "Aggregate the gathered results into the final answer.")
        _emit("plan", model=self.active_model, steps=added)
        return plan

    def synthesize(self, objective: str, plan: Plan, memory: WorkingMemory) -> Optional[str]:
        """Produce the FINAL answer from the full set of gathered tool results.

        Runs once at the end of the loop with the *untruncated* tool outputs, and
        is explicitly instructed to obey any output-format constraint in the
        objective (e.g. "the output must only be a number"). Returns the final
        answer string, or ``None`` to let the engine fall back.
        """
        results = self._tool_results_full(memory)
        if not results:
            return None
        system = (
            "You are Davinci. Produce the FINAL answer to the user's objective using "
            "ONLY the gathered tool results below. Think over the data, do any final "
            "arithmetic needed, and output the result. CRITICAL: obey the objective's "
            "output-format constraint EXACTLY — if it says the output must only be a "
            "number, your \"final_answer\" must be just that number as a string and "
            "nothing else (no words, units, or punctuation). Reply with a single JSON "
            "object: {\"final_answer\": <string>}."
        )
        prompt = json.dumps({
            "objective": objective,
            "tool_results": results,
            "steps": [{"title": s.title, "status": s.status.value, "result": s.result}
                      for s in plan.steps],
        }, indent=2)
        text = self._generate(prompt, system=system)
        decision = self._parse_json(text) if text else None
        if not isinstance(decision, dict):
            _emit("synthesize_fallback")
            return None
        answer = decision.get("final_answer")
        if answer is None:
            return None
        _emit("synthesize", model=self.active_model)
        return str(answer).strip()

    def _tool_results_full(self, memory: WorkingMemory, *, limit: int = 16,
                           cap: int = 6000) -> List[Any]:
        """The run's tool outputs with generous (near-full) detail for synthesis."""
        items = getattr(memory, "items", []) or []
        out: List[Any] = []
        for entry in items:
            if entry.get("role") != "tool_result":
                continue
            content = entry.get("content")
            if isinstance(content, str):
                out.append(content[:cap])
            else:
                blob = json.dumps(content, default=str)
                out.append(json.loads(blob) if len(blob) <= cap else blob[:cap])
        return out[-limit:]

    def reflect(self, objective: str, plan: Plan, memory: WorkingMemory) -> Reflection:
        # A hard failure always warrants a heuristic-style replan signal; ask the
        # model to judge satisfaction and (when unsatisfied) propose next steps.
        system = (
            "You are Davinci's reflection critic. Given the objective and the executed "
            "plan, decide whether the objective is satisfied. Reply with a single JSON "
            "object: {\"satisfied\": bool, \"critique\": <short string>, "
            "\"replan_titles\": [<step titles to retry/add>]}."
        )
        prompt = json.dumps({
            "objective": objective,
            "plan": plan.to_dict(),
            "tool_results": self._memory_digest(memory, roles=("tool_result",)),
        }, indent=2)
        text = self._generate(prompt, system=system)
        decision = self._parse_json(text) if text else None
        if not decision:
            _emit("reflect_fallback")
            return self._fallback.reflect(objective, plan, memory)

        satisfied = bool(decision.get("satisfied")) and not plan.has_failures
        titles = decision.get("replan_titles") or []
        titles = [str(t) for t in titles if str(t).strip()][:4]
        if not satisfied and not titles and plan.has_failures:
            titles = [f"Retry: {s.title}" for s in plan.steps
                      if s.status == StepStatus.FAILED]
        _emit("reflect", model=self.active_model, satisfied=satisfied)
        return Reflection(satisfied=satisfied,
                          critique=str(decision.get("critique") or ""),
                          replan_titles=titles)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _attachments_from_memory(memory: WorkingMemory) -> List[Attachment]:
        """Recover the run's :class:`Attachment` list from working memory."""
        atts = memory.get_fact("attachments", []) if hasattr(memory, "get_fact") else []
        return [a for a in (atts or []) if isinstance(a, Attachment)]

    @staticmethod
    def _memory_digest(memory: WorkingMemory, *, roles: tuple = (),
                       limit: int = 8, cap: int = 1500) -> List[Dict[str, str]]:
        items = getattr(memory, "items", []) or []
        out: List[Dict[str, str]] = []
        for entry in items[-limit:]:
            role = entry.get("role", "")
            if roles and role not in roles:
                continue
            content = entry.get("content")
            text = content if isinstance(content, str) else json.dumps(content, default=str)
            out.append({"role": role, "content": text[:cap]})
        return out


def _emit(event: str, **data: Any) -> None:
    """Greppable, secret-free trace line for the supervising harness."""
    try:
        print("GEMINI " + json.dumps({"event": event, **data}), flush=True)
    except Exception:  # pragma: no cover — never let tracing break the loop
        pass


def reasoner_from_config(config: Dict[str, Any]) -> Optional[GeminiReasoner]:
    """Construct a GeminiReasoner from a config/env, or ``None`` if no key.

    The API key is read from the live ``config.json`` (key ``gemini_api_key`` or
    nested ``gemini.api_key``) or the ``GEMINI_API_KEY`` env var — it is never
    hard-coded or committed.
    """
    gemini = config.get("gemini") if isinstance(config.get("gemini"), dict) else {}
    api_key = (config.get("gemini_api_key") or gemini.get("api_key")
               or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None
    models = (config.get("gemini_models") or gemini.get("models")
              or ([os.environ["GEMINI_MODEL"]] if os.environ.get("GEMINI_MODEL") else None))
    return GeminiReasoner(api_key, models=models)
