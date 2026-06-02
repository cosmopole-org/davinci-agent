"""Observability for Davinci: structured tracing, secret masking, and budgeting.

Every meaningful decision the agent makes is recorded as a structured trace
event so a run can be replayed, audited, and costed after the fact. This
mirrors the trajectory-logging / tracing patterns used by OpenAI Agents SDK,
LangSmith, and OpenHands.

The module is intentionally dependency-free (stdlib only) so it runs inside a
slim Docker creature with no extra packages.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Secret masking
# ---------------------------------------------------------------------------

# Patterns that look like credentials. Masking happens on every event before it
# is persisted or printed, so a leaked token never lands in a transcript.
_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|authorization|bearer)"
               r"\s*[=:]\s*['\"]?([A-Za-z0-9_\-\.]{6,})"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),                 # OpenAI-style keys
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),                # GitHub PAT
    re.compile(r"AKIA[0-9A-Z]{16}"),                    # AWS access key id
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
]

_MASK = "***REDACTED***"


def mask_secrets(value: Any) -> Any:
    """Recursively redact anything that looks like a credential."""
    if isinstance(value, str):
        out = value
        for pat in _SECRET_PATTERNS:
            if pat.groups >= 2:
                out = pat.sub(lambda m: m.group(0).replace(m.group(2), _MASK), out)
            else:
                out = pat.sub(_MASK, out)
        return out
    if isinstance(value, dict):
        return {k: (_MASK if _is_secret_key(k) else mask_secrets(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [mask_secrets(v) for v in value]
    return value


def _is_secret_key(key: str) -> bool:
    lowered = str(key).lower()
    return any(tok in lowered for tok in ("secret", "token", "password", "passwd",
                                          "api_key", "apikey", "private_key", "privatekey"))


# ---------------------------------------------------------------------------
# Token / cost budget
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """Bounded-execution + cost ceiling for one agent run.

    Implements three of the guardrail patterns from the 2026 agent survey:
    bounded execution (max steps / tool calls), token budgeting, and a wall
    clock deadline.
    """

    max_steps: int = 25
    max_tool_calls: int = 100
    max_tokens: int = 200_000
    max_wall_seconds: float = 900.0
    usd_per_1k_tokens: float = 0.0

    steps_used: int = 0
    tool_calls_used: int = 0
    tokens_used: int = 0
    _started_at: float = field(default_factory=time.monotonic)

    def charge_step(self) -> None:
        self.steps_used += 1

    def charge_tool_call(self) -> None:
        self.tool_calls_used += 1

    def charge_tokens(self, n: int) -> None:
        self.tokens_used += max(0, int(n))

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    @property
    def estimated_usd(self) -> float:
        return round(self.tokens_used / 1000.0 * self.usd_per_1k_tokens, 6)

    def exceeded(self) -> Optional[str]:
        """Return the name of the first exhausted limit, or None if within budget."""
        if self.steps_used >= self.max_steps:
            return "max_steps"
        if self.tool_calls_used >= self.max_tool_calls:
            return "max_tool_calls"
        if self.tokens_used >= self.max_tokens:
            return "max_tokens"
        if self.elapsed_seconds >= self.max_wall_seconds:
            return "max_wall_seconds"
        return None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "steps_used": self.steps_used,
            "max_steps": self.max_steps,
            "tool_calls_used": self.tool_calls_used,
            "max_tool_calls": self.max_tool_calls,
            "tokens_used": self.tokens_used,
            "max_tokens": self.max_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "max_wall_seconds": self.max_wall_seconds,
            "estimated_usd": self.estimated_usd,
        }


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceEvent:
    """One immutable, append-only record in a run's trajectory."""

    seq: int
    event_id: str
    ts_utc: str
    kind: str                 # e.g. plan, reason, tool_call, tool_result, decision, error
    message: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Tracer:
    """Thread-safe, append-only tracer.

    Events are masked on the way in, kept in memory for replay, and optionally
    streamed to a sink (stdout JSONL by default) so a supervising process can
    follow the trajectory live.
    """

    def __init__(self, run_id: Optional[str] = None,
                 sink: Optional[Callable[[TraceEvent], None]] = None,
                 stream: bool = False) -> None:
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self._events: List[TraceEvent] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._sink = sink
        self._stream = stream

    def emit(self, kind: str, message: str, **data: Any) -> TraceEvent:
        with self._lock:
            self._seq += 1
            event = TraceEvent(
                seq=self._seq,
                event_id=uuid.uuid4().hex[:12],
                ts_utc=_utc_now(),
                kind=kind,
                message=mask_secrets(message),
                data=mask_secrets(data),
            )
            self._events.append(event)
        if self._stream:
            sys.stdout.write("DAVINCI_TRACE " + json.dumps(event.to_dict()) + "\n")
            sys.stdout.flush()
        if self._sink:
            try:
                self._sink(event)
            except Exception:  # a broken sink must never kill the agent
                pass
        return event

    @property
    def events(self) -> List[TraceEvent]:
        with self._lock:
            return list(self._events)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict()) for e in self.events)

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for e in self.events:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        return {"run_id": self.run_id, "event_count": len(self.events), "by_kind": counts}
