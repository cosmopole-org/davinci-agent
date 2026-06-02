"""Planning + TODO tracking for Davinci.

Davinci is plan-first: it decomposes an objective into ordered, individually
trackable steps before acting (the plan-and-execute pattern), and can replan
when a step fails. A ``Plan`` is a first-class, serialisable object so it can
be surfaced for human review (plan mode / HITL) before any mutation happens.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    id: int
    title: str
    category: str = "general"
    rationale: str = ""
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["status"] = self.status.value
        return d


@dataclass
class Plan:
    objective: str
    steps: List[PlanStep] = field(default_factory=list)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    revision: int = 0
    _id_counter: Any = field(default_factory=lambda: itertools.count(1), repr=False)

    # -- construction --------------------------------------------------------
    def add_step(self, title: str, category: str = "general", rationale: str = "") -> PlanStep:
        step = PlanStep(id=next(self._id_counter), title=title, category=category, rationale=rationale)
        self.steps.append(step)
        return step

    # -- queries -------------------------------------------------------------
    def next_pending(self) -> Optional[PlanStep]:
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    @property
    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps) and bool(self.steps)

    @property
    def has_failures(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def progress(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in StepStatus}
        for step in self.steps:
            counts[step.status.value] += 1
        counts["total"] = len(self.steps)
        return counts

    # -- mutation ------------------------------------------------------------
    def start(self, step_id: int) -> None:
        self._get(step_id).status = StepStatus.IN_PROGRESS

    def complete(self, step_id: int, result: str = "") -> None:
        step = self._get(step_id)
        step.status = StepStatus.DONE
        step.result = result

    def fail(self, step_id: int, result: str = "") -> None:
        step = self._get(step_id)
        step.status = StepStatus.FAILED
        step.result = result

    def replan_failed(self, new_titles: List[str]) -> None:
        """Mark failed steps skipped and append replacement steps."""
        for step in self.steps:
            if step.status == StepStatus.FAILED:
                step.status = StepStatus.SKIPPED
        for title in new_titles:
            self.add_step(title, category="replan", rationale="Inserted by replan-on-failure")
        self.revision += 1

    def _get(self, step_id: int) -> PlanStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"no step with id {step_id}")

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "created_at_utc": self.created_at_utc,
            "revision": self.revision,
            "progress": self.progress(),
            "steps": [s.to_dict() for s in self.steps],
        }

    def render(self) -> str:
        glyph = {
            StepStatus.PENDING: "[ ]",
            StepStatus.IN_PROGRESS: "[~]",
            StepStatus.DONE: "[x]",
            StepStatus.FAILED: "[!]",
            StepStatus.SKIPPED: "[-]",
        }
        lines = [f"Plan for: {self.objective} (rev {self.revision})"]
        for s in self.steps:
            lines.append(f"  {glyph[s.status]} {s.id}. {s.title}  <{s.category}>")
        return "\n".join(lines)


class Planner:
    """Decomposes an objective into a plan.

    The decomposition strategy is pluggable. The default is a deterministic,
    capability-aware heuristic so Davinci can plan with zero external
    dependencies; a real deployment can inject an LLM-backed strategy.
    """

    def __init__(self, strategy: Optional[Any] = None) -> None:
        self.strategy = strategy

    def plan(self, objective: str, required_categories: Optional[List[str]] = None) -> Plan:
        if self.strategy is not None:
            return self.strategy(objective, required_categories or [])
        return self._heuristic_plan(objective, required_categories or [])

    @staticmethod
    def _heuristic_plan(objective: str, required_categories: List[str]) -> Plan:
        plan = Plan(objective=objective)
        plan.add_step("Clarify objective and success criteria", "analysis",
                      "Establish what 'done' means before acting.")
        if required_categories:
            for category in required_categories:
                plan.add_step(f"Execute '{category}' work via the best-matching tool",
                              category, f"Objective requires the {category} capability.")
        else:
            plan.add_step("Select and invoke the appropriate tool(s)", "execution",
                          "Route the task to a registered capability.")
        plan.add_step("Verify results against success criteria", "verification",
                      "Self-check before reporting (reflection).")
        plan.add_step("Synthesize a final answer", "synthesis",
                      "Combine step results into a coherent response.")
        return plan
