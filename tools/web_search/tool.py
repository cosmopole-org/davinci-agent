"""Dedicated tool container entrypoint for web_search."""

from __future__ import annotations

import json


def invoke(payload: dict) -> dict:
    return {"tool_id": "web_search", "status": "ok", "payload": payload}


if __name__ == "__main__":
    print(json.dumps(invoke({"mode": "standalone"})))
