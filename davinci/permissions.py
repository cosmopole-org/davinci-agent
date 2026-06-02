"""Permission, risk, and guardrail layer for Davinci.

This implements the enterprise-grade controls highlighted across Claude Code's
seven-layer permission system and OpenHands' SecurityAnalyzer:

* Risk tiering (LOW / MEDIUM / HIGH) decoupled from enforcement.
* Graduated autonomy via permission modes (plan / default / accept_edits /
  auto / dont_ask / bypass).
* Deny-first rule evaluation (an explicit deny always beats an allow).
* Guardrail layering — validators run on both tool *inputs* and *outputs*.
* Bounded execution is enforced by the engine via ``Budget``.

The result of a check is a ``Decision`` describing whether the action is
allowed, denied, or needs human review, plus the reason — never an exception,
so the caller can route it (e.g. escalate to a human) deterministically.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Risk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]


class PermissionMode(str, Enum):
    PLAN = "plan"               # never mutate; everything needs review
    DEFAULT = "default"         # ask for medium+; allow low
    ACCEPT_EDITS = "accept_edits"  # auto-allow low+medium, review high
    AUTO = "auto"               # allow up to the configured risk ceiling
    DONT_ASK = "dont_ask"       # allow all but log loudly
    BYPASS = "bypass"           # allow all, no gating (dangerous)


class Outcome(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REVIEW = "review"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    reason: str
    risk: Risk = Risk.MEDIUM
    rule: Optional[str] = None

    @property
    def allowed(self) -> bool:
        return self.outcome == Outcome.ALLOW

    def to_dict(self) -> Dict[str, Any]:
        return {"outcome": self.outcome.value, "reason": self.reason,
                "risk": self.risk.value, "rule": self.rule}


@dataclass
class ToolAction:
    """The thing being gated: a tool invocation with its arguments."""

    tool_id: str
    category: str
    risk: Risk = Risk.MEDIUM
    args: Dict[str, Any] = field(default_factory=dict)
    requires_network: bool = False


# A validator inspects an action (or a result) and returns an error string to
# block it, or None to allow it. This is the guardrail-layering primitive.
Validator = Callable[[ToolAction], Optional[str]]
ResultValidator = Callable[[ToolAction, Any], Optional[str]]


class PermissionEngine:
    def __init__(self, mode: PermissionMode = PermissionMode.DEFAULT, *,
                 risk_ceiling: Risk = Risk.MEDIUM,
                 allow_rules: Optional[List[str]] = None,
                 deny_rules: Optional[List[str]] = None) -> None:
        self.mode = mode
        self.risk_ceiling = risk_ceiling
        # Rules match against "<category>:<tool_id>" with glob syntax.
        self.allow_rules = list(allow_rules or [])
        self.deny_rules = list(deny_rules or [])
        self._input_validators: List[Validator] = [default_input_validator]
        self._output_validators: List[ResultValidator] = [default_output_validator]

    # -- validator registration ---------------------------------------------
    def add_input_validator(self, fn: Validator) -> None:
        self._input_validators.append(fn)

    def add_output_validator(self, fn: ResultValidator) -> None:
        self._output_validators.append(fn)

    # -- evaluation ----------------------------------------------------------
    def check(self, action: ToolAction) -> Decision:
        target = f"{action.category}:{action.tool_id}"

        # Layer 1 — input guardrails (deny-first; a failed validator blocks).
        for validator in self._input_validators:
            err = validator(action)
            if err:
                return Decision(Outcome.DENY, f"input guardrail: {err}", action.risk, "input_validator")

        # Layer 2 — deny rules beat everything else.
        for rule in self.deny_rules:
            if fnmatch.fnmatch(target, rule):
                return Decision(Outcome.DENY, f"matched deny rule '{rule}'", action.risk, rule)

        # Layer 3 — explicit allow rules short-circuit to ALLOW.
        for rule in self.allow_rules:
            if fnmatch.fnmatch(target, rule):
                return Decision(Outcome.ALLOW, f"matched allow rule '{rule}'", action.risk, rule)

        # Layer 4 — mode + risk gating.
        return self._mode_decision(action)

    def _mode_decision(self, action: ToolAction) -> Decision:
        risk = action.risk
        if self.mode == PermissionMode.BYPASS:
            return Decision(Outcome.ALLOW, "bypass mode", risk, "mode:bypass")
        if self.mode == PermissionMode.DONT_ASK:
            return Decision(Outcome.ALLOW, "dont_ask mode (logged)", risk, "mode:dont_ask")
        if self.mode == PermissionMode.PLAN:
            return Decision(Outcome.REVIEW, "plan mode forbids execution until approved", risk, "mode:plan")
        if self.mode == PermissionMode.AUTO:
            if risk.rank <= self.risk_ceiling.rank:
                return Decision(Outcome.ALLOW, f"auto mode within ceiling {self.risk_ceiling.value}", risk, "mode:auto")
            return Decision(Outcome.REVIEW, f"risk {risk.value} exceeds ceiling {self.risk_ceiling.value}", risk, "mode:auto")
        if self.mode == PermissionMode.ACCEPT_EDITS:
            if risk.rank <= Risk.MEDIUM.rank:
                return Decision(Outcome.ALLOW, "accept_edits allows low/medium", risk, "mode:accept_edits")
            return Decision(Outcome.REVIEW, "high-risk action needs review", risk, "mode:accept_edits")
        # DEFAULT
        if risk == Risk.LOW:
            return Decision(Outcome.ALLOW, "default mode allows low risk", risk, "mode:default")
        return Decision(Outcome.REVIEW, f"{risk.value}-risk action needs approval", risk, "mode:default")

    def validate_result(self, action: ToolAction, result: Any) -> Optional[str]:
        """Run output guardrails; return an error string if the result is rejected."""
        for validator in self._output_validators:
            err = validator(action, result)
            if err:
                return err
        return None


# ---------------------------------------------------------------------------
# Default guardrails
# ---------------------------------------------------------------------------

# Obvious destructive shell patterns that should never run unattended.
_DANGEROUS_PATTERNS = [
    re.compile(r"rm\s+-rf\s+(/|~|\$HOME)(\s|$)"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\}"),         # fork bomb
    re.compile(r"\bmkfs\.\w+\b"),
    re.compile(r"\bdd\b.*\bof=/dev/(sd|nvme|xvd)"),
    re.compile(r">\s*/dev/(sd|nvme|xvd)"),
]


def default_input_validator(action: ToolAction) -> Optional[str]:
    blob = " ".join(str(v) for v in action.args.values())
    for pat in _DANGEROUS_PATTERNS:
        if pat.search(blob):
            return f"blocked dangerous command pattern: {pat.pattern}"
    return None


def default_output_validator(action: ToolAction, result: Any) -> Optional[str]:
    # Cap absurdly large outputs so a runaway tool can't blow the context.
    text = result if isinstance(result, str) else str(result)
    if len(text) > 5_000_000:
        return "tool output exceeds 5 MB hard cap"
    return None
