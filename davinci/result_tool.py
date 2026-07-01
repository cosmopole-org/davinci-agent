"""Terminal *result* tools — how Davinci delivers its FINAL answer.

Rather than depend on a free-text end-of-run synthesis LLM call to re-type a
value the tools already computed (fragile, and it silently fell back to a
status string when the model didn't echo the number), Davinci exposes a small
set of always-registered **terminal** tools:

    result_as_number   value: number
    result_as_text     value: string
    result_as_boolean  value: boolean
    result_as_json     value: object
    result_as_list     value: array

The LLM finishes a task by calling the one whose type matches the objective's
required output. Because each carries a strict JSON-Schema for ``value``,
tool-use is schema-constrained: the model is forced to hand back a correctly
typed answer instead of prose. The engine intercepts the call, coerces
``value`` into the run's final answer, and ends the run — the answer is then
signalled back to the requester over the same API that delivered the task.

This is deliberately orthogonal to the data tools: plugging in a new tool
(weather, SQL, …) needs no change here, because every task funnels its final
answer through these same terminal tools.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .mcp import ToolDescriptor, ToolRegistry

RESULT_CATEGORY = "result"

# tool name -> the JSON type its ``value`` argument must have
RESULT_TYPES: Dict[str, str] = {
    "result_as_number": "number",
    "result_as_text": "string",
    "result_as_boolean": "boolean",
    "result_as_json": "object",
    "result_as_list": "array",
}

_DESCRIPTIONS: Dict[str, str] = {
    "result_as_number": "TERMINAL. Deliver the FINAL answer as a single number. "
                        "Call this when the objective asks for a numeric result; "
                        "put the number in `value`. Ends the run.",
    "result_as_text": "TERMINAL. Deliver the FINAL answer as free text/string. "
                      "Put the answer in `value`. Ends the run.",
    "result_as_boolean": "TERMINAL. Deliver the FINAL answer as a boolean (true/false). "
                         "Put it in `value`. Ends the run.",
    "result_as_json": "TERMINAL. Deliver the FINAL answer as a JSON object. "
                      "Put the object in `value`. Ends the run.",
    "result_as_list": "TERMINAL. Deliver the FINAL answer as a JSON array/list. "
                      "Put the list in `value`. Ends the run.",
}


def _schema_for(json_type: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "value": {"type": json_type,
                      "description": "The final answer, in the exact required output type."}
        },
        "required": ["value"],
    }


def register_result_tools(registry: ToolRegistry) -> ToolRegistry:
    """Register the always-available terminal result tools on ``registry``.

    Idempotent: re-registering simply overwrites. Safe to call on every registry
    regardless of which data tools the task supplied.
    """
    for name, json_type in RESULT_TYPES.items():
        registry.register(ToolDescriptor(
            name=name,
            description=_DESCRIPTIONS[name],
            category=RESULT_CATEGORY,
            risk="low",
            requires_network=False,
            server="builtin",
            terminal=True,
            _schema=_schema_for(json_type),
        ))
    return registry


def _format_number(v: Any) -> Optional[str]:
    if isinstance(v, bool):  # bool is an int subclass — reject it as a number
        return None
    if isinstance(v, (int, float)):
        return str(v)
    try:
        return str(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def coerce_result_value(tool_name: str, args: Any) -> Optional[str]:
    """Turn a terminal tool call's ``args`` into the final-answer string.

    Returns ``None`` when no usable ``value`` was supplied so the caller can
    fail the step gracefully (and fall back to end-of-run synthesis).
    """
    if not isinstance(args, dict):
        return None
    value = args.get("value")
    if value is None:
        return None
    json_type = RESULT_TYPES.get(tool_name, "string")
    if json_type == "number":
        return _format_number(value)
    if json_type == "boolean":
        if isinstance(value, bool):
            return "true" if value else "false"
        s = str(value).strip().lower()
        if s in ("true", "1", "yes"):
            return "true"
        if s in ("false", "0", "no"):
            return "false"
        return None
    if json_type in ("object", "array"):
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return None
    return str(value)
