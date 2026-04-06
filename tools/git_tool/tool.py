"""Unified git tool container entrypoint."""

from __future__ import annotations

import json


def invoke(function_name: str, payload: dict) -> dict:
    return {
        "tool_id": "git_tool",
        "function": function_name,
        "status": "ok",
        "payload": payload,
    }


if __name__ == "__main__":
    print(json.dumps(invoke("status_commit", {"mode": "standalone"})))
