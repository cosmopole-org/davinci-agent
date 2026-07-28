"""The Davinci agent engine.

A professional agentic loop that combines the patterns the 2026 survey calls
out as table stakes for enterprise agents:

* **Plan-and-execute** — decompose first, then act step by step.
* **ReAct** — each step is reason -> act -> observe.
* **Reflection / self-critique** — verify results before reporting.
* **Replan-on-failure** + **stuck/loop detection**.
* **Bounded execution** via :class:`Budget`, **risk-gated permissions**, and
  **guardrail-layered** tool I/O.
* **Event-sourced tracing** for replay and audit.

Everything external is pluggable through two small protocols — ``Reasoner``
(chooses the action for a step) and ``ToolExecutor`` (performs it) — so the same
engine drives an offline deterministic demo, a Caspar-signalling deployment, or
a real LLM backend without code changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .attachments import Attachment
from .memory import EpisodicMemory, InstructionMemory, WorkingMemory
from .mcp import ToolDescriptor, ToolRegistry
from .observability import Budget, Tracer
from .permissions import Decision, Outcome, PermissionEngine, Risk, ToolAction
from .planning import Plan, Planner, PlanStep, StepStatus


# ---------------------------------------------------------------------------
# Pluggable protocols
# ---------------------------------------------------------------------------

# Canonical conversation roles Davinci reasons over. Upstream systems label the
# assistant side "agent" (Decillion) or "assistant" (OpenAI-style); normalize to
# one vocabulary so the reasoner prompts stay consistent across callers.
_ROLE_ALIASES = {
    "user": "user", "human": "user",
    "assistant": "assistant", "agent": "assistant", "ai": "assistant", "bot": "assistant",
    "system": "system", "developer": "system",
}


def _normalize_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Coerce a raw message list into ``[{role, content, agentName?}]``.

    Tolerant of the shapes different callers emit: ``role``/``from``/``sender``
    for the speaker and ``content``/``text``/``message`` for the body. Entries
    without any text are dropped. Never raises — a malformed history must not
    break a run.
    """
    out: List[Dict[str, Any]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        raw_role = str(item.get("role") or item.get("from") or item.get("sender") or "user").lower()
        role = _ROLE_ALIASES.get(raw_role, "user")
        content = item.get("content")
        if content is None:
            content = item.get("text") or item.get("message") or item.get("answer") or ""
        text = content if isinstance(content, str) else str(content)
        text = text.strip()
        if not text:
            continue
        entry: Dict[str, Any] = {"role": role, "content": text}
        name = item.get("agentName") or item.get("agent") or item.get("name")
        if isinstance(name, str) and name.strip():
            entry["agentName"] = name.strip()
        # Group-chat annotations: who the turn was from, who it was directed at,
        # and whether this agent was one of its targets. Preserved so the
        # reasoner can render "who is talking to whom" for the model.
        sender = item.get("from")
        if isinstance(sender, str) and sender.strip():
            entry["from"] = sender.strip()
        to = item.get("to")
        if isinstance(to, list):
            targets = []
            for t in to:
                if isinstance(t, dict):
                    tname = str(t.get("name") or "").strip()
                    if tname:
                        targets.append({"name": tname,
                                        "kind": "user" if t.get("kind") == "user" else "agent"})
            if targets:
                entry["to"] = targets
        if item.get("directedToMe"):
            entry["directedToMe"] = True
        out.append(entry)
    return out


@dataclass
class ActionProposal:
    """What the reasoner wants to do for a step."""

    tool: Optional[ToolDescriptor]
    args: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    final_answer: Optional[str] = None      # set to finish without a tool


@dataclass
class ToolResult:
    ok: bool
    output: Any
    error: str = ""


class Reasoner(Protocol):
    def propose(self, objective: str, step: PlanStep, registry: ToolRegistry,
                memory: WorkingMemory) -> ActionProposal: ...

    def reflect(self, objective: str, plan: Plan, memory: WorkingMemory) -> "Reflection": ...


class ToolExecutor(Protocol):
    def execute(self, tool: ToolDescriptor, action: ToolAction) -> ToolResult: ...


@dataclass
class Reflection:
    satisfied: bool
    critique: str = ""
    replan_titles: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default offline reasoner / executor (deterministic, no LLM required)
# ---------------------------------------------------------------------------

class HeuristicReasoner:
    """Deterministic reasoner: map a step's category to the best registered tool."""

    def propose(self, objective: str, step: PlanStep, registry: ToolRegistry,
                memory: WorkingMemory) -> ActionProposal:
        if step.category in ("analysis", "synthesis"):
            # Pure-cognition steps don't need a tool.
            answer = None
            if step.category == "synthesis":
                results = [m["content"] for m in memory.items if m["role"] == "tool_result"]
                answer = f"Objective '{objective}' addressed via {len(results)} tool result(s)."
            return ActionProposal(tool=None, thought=f"Cognitive step: {step.title}", final_answer=answer)

        candidates = registry.by_category(step.category) or registry.search(step.title, max_results=1)
        if step.category == "verification":
            return ActionProposal(tool=None, thought="Self-verifying prior results.")
        tool = sorted(candidates, key=lambda t: (Risk(t.risk).rank, t.name))[0] if candidates else None
        return ActionProposal(tool=tool, args={"task": objective, "step": step.title},
                              thought=f"Route '{step.category}' to {tool.name if tool else 'no tool'}")

    def reflect(self, objective: str, plan: Plan, memory: WorkingMemory) -> Reflection:
        if plan.has_failures:
            failed = [s.title for s in plan.steps if s.status == StepStatus.FAILED]
            return Reflection(satisfied=False, critique=f"{len(failed)} step(s) failed",
                              replan_titles=[f"Retry: {t}" for t in failed])
        return Reflection(satisfied=True, critique="All steps completed within guardrails.")


class EchoExecutor:
    """Offline executor that records the intended signal without a backend.

    Useful for tests and dry-runs: it proves the full loop (plan -> permit ->
    execute -> validate -> reflect) without needing a live Caspar node.
    """

    def execute(self, tool: ToolDescriptor, action: ToolAction) -> ToolResult:
        return ToolResult(ok=True, output={"simulated": True, "tool": tool.name, "args": action.args})


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    objective: str
    success: bool
    answer: str
    plan: Dict[str, Any]
    trace: Dict[str, Any]
    budget: Dict[str, Any]
    pending_review: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DavinciAgent:
    def __init__(self, *,
                 registry: Optional[ToolRegistry] = None,
                 permissions: Optional[PermissionEngine] = None,
                 reasoner: Optional[Reasoner] = None,
                 executor: Optional[ToolExecutor] = None,
                 planner: Optional[Planner] = None,
                 tracer: Optional[Tracer] = None,
                 budget: Optional[Budget] = None,
                 instructions: Optional[InstructionMemory] = None,
                 episodic: Optional[EpisodicMemory] = None,
                 max_step_attempts: int = 2) -> None:
        self.registry = registry or ToolRegistry()
        self.permissions = permissions or PermissionEngine()
        self.reasoner = reasoner or HeuristicReasoner()
        self.executor = executor or EchoExecutor()
        self.planner = planner or Planner()
        self.tracer = tracer or Tracer()
        self.budget = budget or Budget()
        self.instructions = instructions or InstructionMemory()
        self.episodic = episodic or EpisodicMemory()
        self.working = WorkingMemory()
        self.max_step_attempts = max_step_attempts
        self._action_fingerprints: Dict[str, int] = {}

    # -- public API ----------------------------------------------------------
    def run(self, objective: str, required_categories: Optional[List[str]] = None,
            replan_budget: int = 1,
            attachments: Optional[List[Attachment]] = None,
            history: Optional[List[Dict[str, Any]]] = None) -> RunResult:
        # Multimodal attachments accompanying the user prompt: surface them on
        # the working memory so reasoners that understand multimodal input
        # (e.g. LLMReasoner) can attach them to their LLM calls. The
        # deterministic HeuristicReasoner simply ignores them.
        atts = list(attachments or [])
        # Set when a step's reasoning delivers the final answer directly (a
        # trivial/conversational turn), so the run loop stops immediately and
        # end-of-run synthesis is skipped rather than regenerating over it.
        self._final_delivered = False
        if atts:
            self.working.set_fact("attachments", atts)
            self.working.remember("user_attachments",
                                  [a.to_dict() for a in atts])
        # Prior conversation for this session (bound to a Decillion space +
        # target agent upstream). Seeding it makes Davinci conversation-based:
        # the reasoner sees earlier turns and can resolve references like "it" /
        # "that" and answer follow-ups coherently instead of one-shot.
        convo = _normalize_history(history)
        if convo:
            self.working.set_fact("conversation", convo)
            for msg in convo:
                self.working.remember(msg["role"], msg["content"])
            # Hand the conversation to reasoners that plan/synthesize with it.
            setter = getattr(self.reasoner, "set_conversation", None)
            if callable(setter):
                setter(convo)
        # Hand the agent's system instruction (its deployed skill / personality)
        # to reasoners that build their own LLM system prompt. Without this the
        # skill is only recorded in the trace below and never reaches the model,
        # so a deployed agent silently ignores the persona it was created with.
        rendered_instructions = self.instructions.render()
        instr_setter = getattr(self.reasoner, "set_instructions", None)
        if callable(instr_setter):
            instr_setter(rendered_instructions)
        self._record("run_start", f"Starting objective: {objective}",
                     instructions=rendered_instructions or None,
                     attachments=[a.to_dict() for a in atts] or None,
                     history_len=len(convo) or None)
        plan = self.planner.plan(objective, required_categories)
        self._record("plan", "Generated execution plan", plan=plan.to_dict())

        answer = ""
        pending_review: List[Dict[str, Any]] = []
        replans_left = replan_budget

        while True:
            limit = self.budget.exceeded()
            if limit:
                self._record("budget_exceeded", f"Stopping: {limit} reached", budget=self.budget.snapshot())
                break

            step = plan.next_pending()
            if step is None:
                # Plan exhausted — reflect, possibly replan.
                reflection = self.reasoner.reflect(objective, plan, self.working)
                self._record("reflection", reflection.critique, satisfied=reflection.satisfied)
                if reflection.satisfied or replans_left <= 0 or not reflection.replan_titles:
                    break
                plan.replan_failed(reflection.replan_titles)
                replans_left -= 1
                self._record("replan", "Replanned after failures", plan=plan.to_dict())
                continue

            answer = self._run_step(objective, plan, step, pending_review) or answer
            if self._final_delivered:
                # A step's reasoning produced the final answer directly.
                self._record("final_answer", "Final answer delivered by step reasoning")
                break

        success = plan.is_complete and not plan.has_failures and self.budget.exceeded() is None
        # Resolve the final answer. There is no terminal result tool: the answer
        # is the LLM's own conversational output, sent straight back over the
        # response signal. If a step already answered directly, keep it; otherwise
        # ask the reasoner to synthesise the final, user-facing answer over the
        # whole run (it also answers purely conversational prompts directly, in
        # persona, instead of forcing tool talk). The deterministic step summary
        # is only a last resort when no LLM reasoner is wired, so a plain
        # "How are you?" never surfaces as "Completed N steps".
        if self._final_delivered and isinstance(answer, str) and answer.strip():
            pass
        else:
            final = self._final_answer(objective, plan)
            answer = final or answer or self._synthesize(objective, plan)
        self._record("run_end", "Run complete", success=success, answer=answer)
        return RunResult(
            objective=objective,
            success=success,
            answer=answer,
            plan=plan.to_dict(),
            trace=self.tracer.summary(),
            budget=self.budget.snapshot(),
            pending_review=pending_review,
        )

    # -- step execution ------------------------------------------------------
    def _run_step(self, objective: str, plan: Plan, step: PlanStep,
                  pending_review: List[Dict[str, Any]]) -> Optional[str]:
        self.budget.charge_step()
        plan.start(step.id)
        step.attempts += 1
        proposal = self.reasoner.propose(objective, step, self.registry, self.working)
        self._record("reason", proposal.thought or step.title, step_id=step.id, category=step.category)

        if proposal.final_answer is not None:
            plan.complete(step.id, proposal.final_answer)
            self.working.remember("assistant", proposal.final_answer)
            # Treat this as the run's final answer only when no further steps
            # remain — so an intermediate cognition step that emits a summary
            # doesn't cut a multi-step plan short. When it IS the last step, the
            # model answered directly (a trivial/conversational turn): keep it and
            # skip redundant end-of-run synthesis.
            if plan.next_pending() is None:
                self._final_delivered = True
            return proposal.final_answer

        if proposal.tool is None:
            # Cognitive / verification / synthesis step with no tool — mark done.
            # The final answer is produced by end-of-run synthesis.
            plan.complete(step.id, "(no tool required)")
            return None

        tool = proposal.tool
        action = ToolAction(tool_id=tool.name, category=tool.category,
                            risk=Risk(tool.risk), args=proposal.args,
                            requires_network=tool.requires_network)

        # Stuck / loop detection.
        fp = self._fingerprint(action)
        self._action_fingerprints[fp] = self._action_fingerprints.get(fp, 0) + 1
        if self._action_fingerprints[fp] > self.max_step_attempts:
            plan.fail(step.id, "stuck: identical action repeated")
            self._record("stuck", "Loop detected; failing step", step_id=step.id, tool=tool.name)
            return None

        decision = self.permissions.check(action)
        self._record("decision", decision.reason, step_id=step.id, tool=tool.name,
                     decision=decision.to_dict())

        if decision.outcome == Outcome.DENY:
            plan.fail(step.id, f"denied: {decision.reason}")
            return None
        if decision.outcome == Outcome.REVIEW:
            pending_review.append({"step_id": step.id, "tool": tool.name,
                                   "args": action.args, "reason": decision.reason,
                                   "risk": decision.risk.value})
            plan.complete(step.id, "(awaiting human review)")
            self._record("human_review", "Action escalated for approval", step_id=step.id, tool=tool.name)
            return None

        # ALLOW — execute through the pluggable backend.
        self.budget.charge_tool_call()
        try:
            result = self.executor.execute(tool, action)
        except Exception as exc:  # executor failures are recoverable steps
            plan.fail(step.id, f"executor error: {exc}")
            self._record("error", f"Tool {tool.name} raised: {exc}", step_id=step.id)
            return None

        if not result.ok:
            plan.fail(step.id, f"tool error: {result.error}")
            self._record("tool_result", f"{tool.name} failed: {result.error}", step_id=step.id, ok=False)
            return None

        out_err = self.permissions.validate_result(action, result.output)
        if out_err:
            plan.fail(step.id, f"output guardrail: {out_err}")
            self._record("guardrail_block", out_err, step_id=step.id, tool=tool.name)
            return None

        self.budget.charge_tokens(self._estimate_tokens(result.output))
        self.working.remember("tool_result", result.output)
        plan.complete(step.id, "ok")
        self._record("tool_result", f"{tool.name} ok", step_id=step.id, ok=True, output=result.output)
        return None

    # -- helpers -------------------------------------------------------------
    def _final_answer(self, objective: str, plan: Plan) -> Optional[str]:
        """Ask the reasoner to synthesise the final answer from all tool results.

        Optional on the Reasoner protocol — the deterministic HeuristicReasoner
        does not implement it, so this returns ``None`` there and the engine
        falls back to its summary.
        """
        synth = getattr(self.reasoner, "synthesize", None)
        if not callable(synth):
            return None
        try:
            answer = synth(objective, plan, self.working)
        except Exception as exc:  # noqa: BLE001 — never fail the run on synthesis
            self._record("error", f"synthesis failed: {exc}")
            return None
        return answer or None

    def _synthesize(self, objective: str, plan: Plan) -> str:
        done = sum(1 for s in plan.steps if s.status == StepStatus.DONE)
        return (f"Completed {done}/{len(plan.steps)} planned steps for objective "
                f"'{objective}'. Failures: {plan.has_failures}.")

    def _fingerprint(self, action: ToolAction) -> str:
        raw = f"{action.tool_id}|{sorted(action.args.items())}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _estimate_tokens(value: Any) -> int:
        text = value if isinstance(value, str) else str(value)
        return max(1, len(text) // 4)  # ~4 chars/token heuristic

    def _record(self, kind: str, message: str, **data: Any) -> None:
        event = self.tracer.emit(kind, message, **data)
        self.episodic.append(event.to_dict())
