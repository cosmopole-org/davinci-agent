"""Layered memory for Davinci.

Three layers, matching the memory taxonomy in the 2026 agent survey:

1. ``WorkingMemory``      — short-term scratchpad for the current run
   (observations, intermediate results, the rolling message window).
2. ``EpisodicMemory``     — an append-only JSONL event log per run, enabling
   deterministic replay / resume / audit (OpenHands-style event sourcing).
3. ``InstructionMemory``  — hierarchical, persistent guidance loaded from
   ``DAVINCI.md`` files (managed -> project -> directory), the analogue of
   Claude Code's ``CLAUDE.md`` memory hierarchy.

A lightweight context-compaction helper implements the "cheapest reduction
first" idea: cap per-item size, then drop oldest, then summarise.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Working memory
# ---------------------------------------------------------------------------

class WorkingMemory:
    """Bounded short-term store for the active run."""

    def __init__(self, max_items: int = 200) -> None:
        self.max_items = max_items
        self._items: List[Dict[str, Any]] = []
        self._facts: Dict[str, Any] = {}

    def remember(self, role: str, content: Any) -> None:
        self._items.append({"role": role, "content": content})
        if len(self._items) > self.max_items:
            # Keep the oldest (likely the goal) plus the most recent window.
            self._items = self._items[:1] + self._items[-(self.max_items - 1):]

    def set_fact(self, key: str, value: Any) -> None:
        self._facts[key] = value

    def get_fact(self, key: str, default: Any = None) -> Any:
        return self._facts.get(key, default)

    @property
    def items(self) -> List[Dict[str, Any]]:
        return list(self._items)

    @property
    def facts(self) -> Dict[str, Any]:
        return dict(self._facts)


# ---------------------------------------------------------------------------
# Episodic memory (event-sourced JSONL log)
# ---------------------------------------------------------------------------

@dataclass
class EpisodicMemory:
    """Append-only event log; the durable record of a run."""

    path: Optional[str] = None
    _events: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, event: Dict[str, Any]) -> None:
        record = dict(event)
        record.setdefault("ts_utc", datetime.now(timezone.utc).isoformat())
        self._events.append(record)
        if self.path:
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
            except OSError:
                pass  # never let logging break the run

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    @classmethod
    def replay(cls, path: str) -> "EpisodicMemory":
        """Reconstruct episodic memory from a JSONL file (resume / audit)."""
        mem = cls(path=path)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            mem._events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return mem


# ---------------------------------------------------------------------------
# Instruction memory (DAVINCI.md hierarchy)
# ---------------------------------------------------------------------------

class InstructionMemory:
    """Hierarchical persistent instructions loaded from ``DAVINCI.md`` files.

    Files are merged from least to most specific so a directory-level file can
    override project-level guidance, exactly like the CLAUDE.md hierarchy.
    """

    FILENAME = "DAVINCI.md"

    def __init__(self, sources: Optional[List[str]] = None) -> None:
        self._blocks: List[Dict[str, str]] = []
        for src in sources or []:
            self.load_file(src)

    def load_file(self, path: str) -> bool:
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self._blocks.append({"source": path, "text": fh.read().strip()})
                return True
            except OSError:
                return False
        return False

    def discover(self, start_dir: str, root_dir: Optional[str] = None) -> "InstructionMemory":
        """Walk upward from ``start_dir`` collecting DAVINCI.md files.

        Files closer to ``start_dir`` are appended last so they take priority.
        """
        start = os.path.abspath(start_dir)
        stop = os.path.abspath(root_dir) if root_dir else os.path.abspath(os.sep)
        chain: List[str] = []
        cur = start
        while True:
            candidate = os.path.join(cur, self.FILENAME)
            if os.path.isfile(candidate):
                chain.append(candidate)
            if cur == stop or os.path.dirname(cur) == cur:
                break
            cur = os.path.dirname(cur)
        for path in reversed(chain):  # outermost first, innermost last (wins)
            self.load_file(path)
        return self

    def add_inline(self, text: str, source: str = "inline") -> None:
        if text and text.strip():
            self._blocks.append({"source": source, "text": text.strip()})

    def render(self) -> str:
        if not self._blocks:
            return ""
        parts = [f"## Instructions from {b['source']}\n{b['text']}" for b in self._blocks]
        return "\n\n".join(parts)

    @property
    def blocks(self) -> List[Dict[str, str]]:
        return list(self._blocks)


# ---------------------------------------------------------------------------
# Session memory (per-session conversation)
# ---------------------------------------------------------------------------

class SessionMemory:
    """Bounded, in-process store of the conversation for each session.

    A davinci creature serves many sessions sequentially (one per space +
    target-agent thread upstream). This keeps the rolling message list for each
    so a run has conversational continuity even when the caller does not resend
    the full history. When the caller *does* supply the authoritative history
    (the orchestrator that owns targeting), :meth:`replace` adopts it verbatim.

    State is intentionally in-memory only: the durable, targeting-aware record
    lives in the orchestrator; this is a per-process cache, not the source of
    truth.
    """

    def __init__(self, max_messages: int = 100, max_sessions: int = 512) -> None:
        self.max_messages = max_messages
        self.max_sessions = max_sessions
        self._sessions: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

    def _touch(self, session_id: str) -> List[Dict[str, Any]]:
        msgs = self._sessions.pop(session_id, None)
        if msgs is None:
            msgs = []
        self._sessions[session_id] = msgs  # move-to-end (most-recently-used)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)  # evict least-recently-used
        return msgs

    def messages(self, session_id: str) -> List[Dict[str, Any]]:
        return [dict(m) for m in self._sessions.get(session_id, [])]

    def append(self, session_id: str, role: str, content: str,
               agent_name: Optional[str] = None) -> None:
        text = (content or "").strip()
        if not text:
            return
        msgs = self._touch(session_id)
        entry: Dict[str, Any] = {"role": role, "content": text}
        if agent_name:
            entry["agentName"] = agent_name
        msgs.append(entry)
        if len(msgs) > self.max_messages:
            del msgs[: len(msgs) - self.max_messages]

    def replace(self, session_id: str, history: Iterable[Dict[str, Any]]) -> None:
        """Adopt an authoritative message list for a session (bounded)."""
        cleaned = [
            {k: v for k, v in m.items() if k in ("role", "content", "agentName")}
            for m in (history or [])
            if isinstance(m, dict) and str(m.get("content") or "").strip()
        ]
        self._sessions.pop(session_id, None)
        self._sessions[session_id] = cleaned[-self.max_messages:]
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------

def compact_messages(messages: Iterable[Dict[str, Any]], *, max_chars_per_item: int = 4000,
                     max_items: int = 40) -> List[Dict[str, Any]]:
    """Cheapest-first context reduction.

    1. Cap each item's serialized size (budget reduction).
    2. Keep the first item (the goal) and the most recent window (temporal trim).
    """
    items = list(messages)
    capped: List[Dict[str, Any]] = []
    for it in items:
        content = it.get("content", "")
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        if len(text) > max_chars_per_item:
            half = max_chars_per_item // 2
            text = text[:half] + f"\n…[{len(text) - max_chars_per_item} chars trimmed]…\n" + text[-half:]
        capped.append({"role": it.get("role", "?"), "content": text})
    if len(capped) <= max_items:
        return capped
    return capped[:1] + capped[-(max_items - 1):]
