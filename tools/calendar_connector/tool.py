"""Dedicated tool container entrypoint for calendar_connector."""

from __future__ import annotations

import json


def invoke(payload: dict) -> dict:
    return {"tool_id": "calendar_connector", "status": "ok", "payload": payload}


if __name__ == "__main__":
    print(json.dumps(invoke({"mode": "standalone"})))
