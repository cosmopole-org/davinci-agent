"""Dedicated tool container entrypoint for python_exec."""

from __future__ import annotations

import json


def invoke(payload: dict) -> dict:
    return {"tool_id": "python_exec", "status": "ok", "payload": payload}


if __name__ == "__main__":
    print(json.dumps(invoke({"mode": "standalone"})))
