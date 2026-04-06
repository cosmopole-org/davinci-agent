"""Dedicated tool container entrypoint for fetch_url."""

from __future__ import annotations

import json


def invoke(payload: dict) -> dict:
    return {"tool_id": "fetch_url", "status": "ok", "payload": payload}


if __name__ == "__main__":
    print(json.dumps(invoke({"mode": "standalone"})))
