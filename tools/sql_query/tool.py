"""Dedicated tool container entrypoint for sql_query."""

from __future__ import annotations

import json


def invoke(payload: dict) -> dict:
    return {"tool_id": "sql_query", "status": "ok", "payload": payload}


if __name__ == "__main__":
    print(json.dumps(invoke({"mode": "standalone"})))
