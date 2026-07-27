"""Provider-agnostic LLM reasoner — Davinci's brain, written once.

:class:`LLMReasoner` implements the engine's ``Reasoner`` protocol
(``propose`` + ``reflect``) plus the optional ``make_plan`` and ``synthesize``
hooks. **All of the reasoning logic lives here and is shared across every LLM**:
the planning decomposition, per-step tool selection, reflection/replan, and
end-of-run synthesis — including the prompts, the tolerant JSON parsing, and the
graceful fall-back to the deterministic :class:`HeuristicReasoner`.

It is *entirely* decoupled from any specific provider: it talks to an
:class:`~davinci.llm.LLMClient` through one method — ``client.generate(...)`` —
so the same algorithms drive Gemini, Anthropic (Claude), OpenAI, Grok, or any
OpenAI-compatible endpoint. Switching models never touches this file.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .attachments import Attachment
from .engine import ActionProposal, HeuristicReasoner, Reflection
from .llm import LLMClient, client_from_config
from .mcp import ToolRegistry
from .memory import WorkingMemory
from .planning import Plan, PlanStep, StepStatus


class LLMReasoner:
    """An LLM-backed :class:`Reasoner` parameterised by an :class:`LLMClient`.

    The client is the *only* provider-specific dependency; everything else here
    is shared logic. Construct directly with any client, or via
    :func:`reasoner_from_config`.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self._fallback = HeuristicReasoner()
        # Prior turns of this session's conversation, seeded by the engine at the
        # start of a run (see ``DavinciAgent.run(history=...)``). Empty for a
        # fresh one-shot prompt.
        self._conversation: List[Dict[str, str]] = []
        # The agent's system instruction (its deployed skill / personality),
        # seeded by the engine (see ``DavinciAgent.set_instructions``). Prepended
        # to every LLM system prompt so the model actually adopts the persona —
        # empty for an agent deployed without a skill.
        self._instructions: str = ""

    # -- conversation --------------------------------------------------------
    def set_conversation(self, history: List[Dict[str, Any]]) -> None:
        """Adopt the session's prior messages so planning, per-step reasoning,
        and final synthesis are all conducted with conversational context."""
        self._conversation = [
            {"role": str(m.get("role") or "user"),
             "content": str(m.get("content") or "")}
            for m in (history or [])
            if isinstance(m, dict) and str(m.get("content") or "").strip()
        ]

    # -- system instruction (deployed skill / personality) ------------------
    def set_instructions(self, text: Optional[str]) -> None:
        """Adopt the agent's system instruction (the skill file a Caspar proxy
        "agent" entity attaches, layered on the DAVINCI.md hierarchy).

        The engine hands this over so planning, per-step tool selection,
        reflection and final synthesis are all conditioned on the persona. Without
        it the skill would only appear in the run trace and never reach the model,
        so a deployed agent would ignore the personality it was created with."""
        self._instructions = (text or "").strip()

    def _system(self, base: str) -> str:
        """Prepend the agent's system instruction to a task-specific system
        prompt so the model reasons and answers in the deployed persona. A no-op
        when the agent has no skill.

        The persona is declared *authoritative for identity*: it overrides any
        default name or role in the task-framework text that follows, so an agent
        deployed as "Tina" answers "who are you?" as Tina, not as the internal
        planner. Without this the hardcoded framework identity leaks as the
        agent's self-description."""
        if not self._instructions:
            return base
        return (
            "You ARE the persona defined below. It is your identity, name, voice, "
            "skills and rules, and it OVERRIDES any default name, role or "
            "personality mentioned in the task framework that follows. When asked "
            "who you are or what you do, answer strictly as this persona — never "
            "introduce yourself by any internal engine, planner or framework name.\n\n"
            "=== YOUR PERSONA (system instruction) ===\n"
            f"{self._instructions}\n"
            "=== END PERSONA ===\n\n"
            "What follows describes HOW to carry out the task. Follow its mechanics, "
            "but keep the persona above as your identity and the voice of every "
            "user-facing answer:\n"
            f"{base}"
        )

    def _conversation_digest(self, *, limit: int = 20, cap: int = 2000) -> List[Dict[str, str]]:
        """Recent conversation turns, size-bounded for prompt inclusion."""
        digest: List[Dict[str, str]] = []
        for m in self._conversation[-limit:]:
            text = m.get("content", "")
            digest.append({"role": m.get("role", "user"), "content": text[:cap]})
        return digest

    # -- pass-through identity (used by the runtime for capability reporting) --
    @property
    def provider(self) -> str:
        return self.client.provider

    @property
    def models(self) -> List[str]:
        return self.client.models

    @property
    def active_model(self) -> Optional[str]:
        return self.client.active_model

    def _emit(self, event: str, **data: Any) -> None:
        self.client.emit(event, **data)

    # -- shared JSON helpers ------------------------------------------------
    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Tolerantly parse a JSON object from an LLM reply.

        Strips markdown code fences and, as a last resort, grabs the outermost
        ``{...}`` span — providers escape and wrap JSON differently, so never
        raw-string-match the reply.
        """
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(text[start:end + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
            return None

    # -- Reasoner protocol: per-step tool selection ------------------------
    def propose(self, objective: str, step: PlanStep, registry: ToolRegistry,
                memory: WorkingMemory) -> ActionProposal:
        # The final synthesis is handled once, at end-of-run, with the full
        # untruncated tool results (see ``synthesize``).
        if step.category == "synthesis":
            return ActionProposal(
                tool=None,
                thought="final answer deferred to end-of-run synthesis over full results")
        # Include each tool's input schema so the model emits arguments that
        # match the tool's real contract.
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
            "You are an enterprise agent planner. For the CURRENT step you "
            "pick exactly one registered tool to advance the objective, or no tool for "
            "pure-cognition (analysis/synthesis/verification) steps. Always reply with "
            "a single JSON object: {\"tool\": <tool name or null>, \"args\": {object}, "
            "\"thought\": <short reason>, \"final_answer\": <string or null>}. "
            "Choose the least-risk tool that fits the step's category. CRITICAL: fill "
            "\"args\" using the chosen tool's \"arg_schema\" (use those exact property "
            "names and types) — e.g. provide runnable code for a code arg, a real query "
            "string for a query arg. Only fall back to a generic args.task when the tool "
            "exposes no arg_schema. To DELIVER THE FINAL ANSWER (e.g. a \"result\" step, "
            "or once you have computed the result), pick the terminal tool whose type "
            "matches the required output — result_as_number / result_as_text / "
            "result_as_boolean / result_as_json / result_as_list — and put the answer in "
            "args.value with the correct type; that ends the run. When the user supplied attachments, they are sent "
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
        if self._conversation:
            prompt_payload["conversation"] = self._conversation_digest()
        if attachments:
            prompt_payload["attachments"] = [
                {"name": a.name, "mime_type": a.mime_type, "size": a.size,
                 "kind": a.kind, "description": a.description}
                for a in attachments
            ]
        prompt = json.dumps(prompt_payload, indent=2)

        text = self.client.generate(prompt, system=self._system(system), attachments=attachments)
        decision = self._parse_json(text) if text else None
        if not decision:
            self._emit("propose_fallback", step=step.title)
            return self._fallback.propose(objective, step, registry, memory)

        thought = str(decision.get("thought") or f"[{self.provider}:{self.active_model}] {step.title}")
        final_answer = decision.get("final_answer")
        if isinstance(final_answer, str) and final_answer.strip():
            self._emit("propose", model=self.active_model, step=step.title,
                       decision="final_answer")
            return ActionProposal(tool=None, thought=thought, final_answer=final_answer.strip())

        tool_name = decision.get("tool")
        tool = registry.get(tool_name) if isinstance(tool_name, str) else None
        if tool_name and not tool:
            matches = registry.search(str(tool_name), max_results=1)
            tool = matches[0] if matches else None
        args = decision.get("args") if isinstance(decision.get("args"), dict) else {}
        args.setdefault("task", objective)
        args.setdefault("step", step.title)
        self._emit("propose", model=self.active_model, step=step.title,
                   tool=tool.name if tool else None)
        return ActionProposal(tool=tool, args=args, thought=thought)

    # -- planning strategy --------------------------------------------------
    def make_plan(self, objective: str, required_categories: List[str],
                  registry: ToolRegistry) -> Plan:
        """Decompose the objective into steps — but only as many as it needs.

        Planning is *dynamic*: the model first judges how much planning the
        request actually warrants, then returns the SMALLEST plan that gets it
        done. A trivial or conversational request collapses to a single
        result-delivering step (no analysis/verification/synthesis scaffold);
        only a genuinely multi-part objective gets a full decomposition. This
        keeps Davinci flexible instead of over-planning every prompt.
        """
        from .planning import Planner as _Planner  # local import avoids a cycle

        tools = [t.stub() for t in registry.all()]
        categories = sorted({t.get("category", "general") for t in tools} |
                            {"analysis", "verification", "synthesis"})
        system = (
            "You are the task planner. FIRST judge how much planning the objective "
            "actually needs, then produce the SMALLEST plan that accomplishes it — "
            "never over-plan. Match the plan's shape to the request:\n"
            "- TRIVIAL (a greeting, a direct question you can answer from your own "
            "knowledge, a one-line fact or calculation needing no tools): return a "
            "SINGLE step with category \"result\" whose title states the answer to "
            "deliver. Do NOT add analysis, verification or synthesis steps.\n"
            "- SIMPLE (one tool call or one lookup answers it): return just that one "
            "execution step followed by the final \"result\" step (about 2 steps).\n"
            "- COMPLEX (multi-part; needs research, then computation, then checking): "
            "decompose into a concrete, ordered, minimal sequence (up to ~9 steps), "
            "adding analysis/verification steps ONLY where the difficulty warrants "
            "them.\n"
            "Each step's \"title\" must be a specific, actionable instruction (name the "
            "exact URL, data, or computation — never a vague 'do the research' "
            "placeholder). Pick each step's \"category\" from the provided list so it "
            "routes to a matching tool. The LAST step MUST have category \"result\": it "
            "delivers the final answer by calling the terminal result tool whose type "
            "matches the objective's required output (e.g. result_as_number for a "
            "number), honoring any output-format constraint stated in the objective. "
            "Reply with a single JSON object: {\"complexity\": "
            "\"trivial\"|\"simple\"|\"complex\", \"steps\": [{\"title\": str, "
            "\"category\": str, \"rationale\": str}, ...]}."
        )
        if self._conversation:
            system += (
                " This objective is the latest turn of an ONGOING conversation "
                "(prior turns are provided as \"conversation\"). Interpret it in "
                "that context — resolve references to earlier turns (\"it\", "
                "\"that\", \"the same\") and do not re-do work already answered."
            )
        prompt_payload: Dict[str, Any] = {
            "objective": objective,
            "available_categories": categories,
            "available_tools": tools,
        }
        if self._conversation:
            prompt_payload["conversation"] = self._conversation_digest()
        prompt = json.dumps(prompt_payload, indent=2)
        # Planning is internal machinery, not a user-facing turn: the persona is
        # deliberately NOT injected here so it cannot derail the structured plan
        # (a strong "answer as this persona" directive makes the planner
        # over-decompose or break its JSON). Persona shapes the ANSWER, not the
        # plan. Personality-driven brevity is honoured at synthesis time.
        text = self.client.generate(prompt, system=system)
        decision = self._parse_json(text) if text else None
        steps = (decision or {}).get("steps") if isinstance(decision, dict) else None
        if not isinstance(steps, list) or not steps:
            self._emit("plan_fallback")
            return _Planner._heuristic_plan(objective, required_categories)
        complexity = str((decision or {}).get("complexity") or "").strip().lower()

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
        if plan.steps[-1].category not in ("result", "synthesis"):
            plan.add_step("Deliver the final answer by calling the matching result_as_* "
                          "terminal tool with the computed value, in the exact required format",
                          "result", "Submit the typed final answer so it is signalled back "
                          "to the requester.")
        self._emit("plan", model=self.active_model, steps=added,
                   complexity=complexity or None)
        return plan

    # -- end-of-run synthesis ----------------------------------------------
    def synthesize(self, objective: str, plan: Plan, memory: WorkingMemory) -> Optional[str]:
        """Produce the FINAL, user-facing answer — in the deployed persona's voice.

        This is the agent's single guaranteed chance to *talk to the user*, and it
        handles both halves of a general-purpose agent with no branching in the
        engine:

        * a tool-driven run — synthesise the answer over the gathered results;
        * a purely conversational turn — a greeting, small talk, or a question
          about who/how the agent is — where there are NO tool results: answer it
          directly from the conversation and the agent's own knowledge.

        It must never degrade a conversational turn into a "completed N steps"
        summary. Because it always attempts an answer, the engine can keep one
        uniform path (plan → act → synthesise) for every prompt."""
        results = self._tool_results_full(memory)
        has_results = bool(results)
        source = (
            "the gathered tool results below"
            + (" together with the prior conversation" if self._conversation else "")
        ) if has_results else "the prior conversation and your own knowledge"
        system = (
            f"Produce the FINAL answer to the user's objective using {source}. "
            "If the objective is conversational — a greeting, small talk, or a "
            "question about who you are or how you are — just reply to it naturally "
            "and directly; do NOT mention tools, plans or steps, and never say you "
            "'completed' anything. Otherwise think over the data, do any final "
            "arithmetic needed, and output the result. CRITICAL: obey the "
            "objective's output-format constraint EXACTLY — if it says the output "
            "must only be a number, your \"final_answer\" must be just that number as "
            "a string and nothing else (no words, units, or punctuation). Reply with "
            "a single JSON object: {\"final_answer\": <string>}."
        )
        payload: Dict[str, Any] = {
            "objective": objective,
            "tool_results": results,
            "steps": [{"title": s.title, "status": s.status.value, "result": s.result}
                      for s in plan.steps],
        }
        if self._conversation:
            payload["conversation"] = self._conversation_digest()
        prompt = json.dumps(payload, indent=2)
        text = self.client.generate(prompt, system=self._system(system))
        decision = self._parse_json(text) if text else None
        if not isinstance(decision, dict):
            self._emit("synthesize_fallback")
            return None
        answer = decision.get("final_answer")
        if answer is None:
            return None
        self._emit("synthesize", model=self.active_model)
        return str(answer).strip()

    # -- reflection / replan ------------------------------------------------
    def reflect(self, objective: str, plan: Plan, memory: WorkingMemory) -> Reflection:
        system = (
            "You are the reflection critic. Given the objective and the executed "
            "plan, decide whether the objective is satisfied. Reply with a single JSON "
            "object: {\"satisfied\": bool, \"critique\": <short string>, "
            "\"replan_titles\": [<step titles to retry/add>]}."
        )
        prompt = json.dumps({
            "objective": objective,
            "plan": plan.to_dict(),
            "tool_results": self._memory_digest(memory, roles=("tool_result",)),
        }, indent=2)
        # Reflection is internal machinery too (a satisfied/replan verdict) — keep
        # the persona out of it so it stays a reliable structured critic.
        text = self.client.generate(prompt, system=system)
        decision = self._parse_json(text) if text else None
        if not decision:
            self._emit("reflect_fallback")
            return self._fallback.reflect(objective, plan, memory)

        satisfied = bool(decision.get("satisfied")) and not plan.has_failures
        titles = decision.get("replan_titles") or []
        titles = [str(t) for t in titles if str(t).strip()][:4]
        if not satisfied and not titles and plan.has_failures:
            titles = [f"Retry: {s.title}" for s in plan.steps
                      if s.status == StepStatus.FAILED]
        self._emit("reflect", model=self.active_model, satisfied=satisfied)
        return Reflection(satisfied=satisfied,
                          critique=str(decision.get("critique") or ""),
                          replan_titles=titles)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _attachments_from_memory(memory: WorkingMemory) -> List[Attachment]:
        atts = memory.get_fact("attachments", []) if hasattr(memory, "get_fact") else []
        return [a for a in (atts or []) if isinstance(a, Attachment)]

    @staticmethod
    def _memory_digest(memory: WorkingMemory, *, roles: tuple = (),
                       limit: int = 8, cap: int = 1500,
                       tool_cap: int = 24000) -> List[Dict[str, str]]:
        # Tool results carry the actual gathered data (e.g. a scraped page's
        # rendered text); a tight cap would hide that payload from the model and
        # force it to act on the header/nav only. Give tool_result entries a much
        # larger budget than chat turns so the next step reasons over real data.
        items = getattr(memory, "items", []) or []
        out: List[Dict[str, str]] = []
        for entry in items[-limit:]:
            role = entry.get("role", "")
            if roles and role not in roles:
                continue
            content = entry.get("content")
            text = content if isinstance(content, str) else json.dumps(content, default=str)
            limit_chars = tool_cap if role == "tool_result" else cap
            out.append({"role": role, "content": text[:limit_chars]})
        return out

    @staticmethod
    def _tool_results_full(memory: WorkingMemory, *, limit: int = 16,
                           cap: int = 24000) -> List[Any]:
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


def reasoner_from_config(config: Dict[str, Any]) -> Optional[LLMReasoner]:
    """Construct an :class:`LLMReasoner` for whichever provider the config names.

    Provider/API-key resolution is delegated to
    :func:`davinci.llm.client_from_config` (supports Gemini, Anthropic, OpenAI,
    Grok, or any OpenAI-compatible endpoint). Returns ``None`` when no provider
    has a key, so the caller falls back to the deterministic heuristic reasoner.
    Keys are read from the live config/env only — never hard-coded or committed.
    """
    client = client_from_config(config)
    return LLMReasoner(client) if client is not None else None
