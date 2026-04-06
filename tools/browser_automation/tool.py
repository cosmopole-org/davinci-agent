"""Dedicated tool container entrypoint for browser_automation."""

from __future__ import annotations

import json


def invoke(payload: dict) -> dict:
    return {"tool_id": "browser_automation", "status": "ok", "payload": payload}


if __name__ == "__main__":
    print(json.dumps(invoke({"mode": "standalone"})))
