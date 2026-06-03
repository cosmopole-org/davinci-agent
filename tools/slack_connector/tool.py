"""slack_connector tool creature — real Slack Web API operations.

Uses the official ``slack_sdk`` WebClient. Authentication via
``SLACK_BOT_TOKEN`` (or ``payload['token']``).

Signal payload (function ``slack_action`` or ``invoke``)::

    {
      "operation": "post_message",   # see table below
      "channel": "#general",          # name (resolved) or id
      "text": "hello",
      ...operation-specific fields...
    }

Operations:
    post_message   {channel, text, blocks?, thread_ts?}
    update_message {channel, ts, text}
    delete_message {channel, ts}
    reply          {channel, thread_ts, text}
    add_reaction   {channel, ts, name}
    list_channels  {types?, limit?}
    history        {channel, limit?}
    upload_file    {channel, content|file_path, filename?, title?, initial_comment?}
    users_list     {limit?}
    find_channel   {channel}
    auth_test      {}
"""

from __future__ import annotations

import json
import os


def _client(payload: dict):
    from slack_sdk import WebClient
    token = payload.get("token") or os.environ.get("SLACK_BOT_TOKEN") or os.environ.get("SLACK_TOKEN")
    if not token:
        raise RuntimeError("no Slack token (set SLACK_BOT_TOKEN or payload.token)")
    return WebClient(token=token)


def _resolve_channel(client, channel: str) -> str:
    if not channel or not channel.startswith("#"):
        return channel
    name = channel.lstrip("#")
    cursor = None
    for _ in range(20):
        resp = client.conversations_list(limit=1000, cursor=cursor,
                                         types="public_channel,private_channel")
        for ch in resp["channels"]:
            if ch.get("name") == name:
                return ch["id"]
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return channel


def _op(client, op: str, p: dict) -> dict:
    if op == "auth_test":
        return dict(client.auth_test().data)

    if op in ("post_message", "send_message", "message", "reply"):
        ch = _resolve_channel(client, p.get("channel"))
        kwargs = {"channel": ch, "text": p.get("text", "")}
        if p.get("blocks"):
            kwargs["blocks"] = p["blocks"]
        if p.get("thread_ts"):
            kwargs["thread_ts"] = p["thread_ts"]
        r = client.chat_postMessage(**kwargs)
        return {"channel": r["channel"], "ts": r["ts"], "message": r["message"].get("text")}

    if op == "update_message":
        ch = _resolve_channel(client, p.get("channel"))
        r = client.chat_update(channel=ch, ts=p["ts"], text=p.get("text", ""))
        return {"channel": r["channel"], "ts": r["ts"]}

    if op == "delete_message":
        ch = _resolve_channel(client, p.get("channel"))
        r = client.chat_delete(channel=ch, ts=p["ts"])
        return {"channel": r["channel"], "ts": r["ts"], "deleted": True}

    if op == "add_reaction":
        ch = _resolve_channel(client, p.get("channel"))
        client.reactions_add(channel=ch, timestamp=p["ts"], name=p.get("name", "thumbsup"))
        return {"reacted": True, "name": p.get("name", "thumbsup")}

    if op in ("list_channels", "conversations_list"):
        r = client.conversations_list(limit=int(p.get("limit", 200)),
                                      types=p.get("types", "public_channel,private_channel"))
        return {"channels": [{"id": c["id"], "name": c.get("name"),
                              "is_private": c.get("is_private"),
                              "num_members": c.get("num_members")} for c in r["channels"]]}

    if op in ("history", "conversations_history"):
        ch = _resolve_channel(client, p.get("channel"))
        r = client.conversations_history(channel=ch, limit=int(p.get("limit", 50)))
        return {"channel": ch, "messages": [{"ts": m.get("ts"), "user": m.get("user"),
                                             "text": m.get("text")} for m in r["messages"]]}

    if op in ("upload_file", "files_upload"):
        ch = _resolve_channel(client, p.get("channel"))
        kwargs = {"channel": ch, "filename": p.get("filename", "upload.txt"),
                  "title": p.get("title"), "initial_comment": p.get("initial_comment")}
        if p.get("file_path"):
            kwargs["file"] = p["file_path"]
        else:
            kwargs["content"] = p.get("content", "")
        r = client.files_upload_v2(**{k: v for k, v in kwargs.items() if v is not None})
        files = r.get("files") or ([r.get("file")] if r.get("file") else [])
        return {"uploaded": True, "files": [{"id": f.get("id"), "name": f.get("name")}
                                            for f in files if f]}

    if op in ("users_list", "list_users"):
        r = client.users_list(limit=int(p.get("limit", 200)))
        return {"members": [{"id": u["id"], "name": u.get("name"),
                             "real_name": u.get("real_name"),
                             "is_bot": u.get("is_bot")} for u in r["members"]]}

    if op == "find_channel":
        return {"channel_id": _resolve_channel(client, p.get("channel"))}

    raise ValueError(f"unknown slack operation '{op}'")


def invoke(function_name: str, payload: dict) -> dict:
    payload = payload or {}
    op = payload.get("operation") or payload.get("action") or "post_message"
    try:
        client = _client(payload)
        data = _op(client, op, payload)
        return {"ok": True, "operation": op, "function": function_name, "result": data}
    except Exception as exc:
        # SlackApiError carries a useful .response
        resp = getattr(exc, "response", None)
        detail = None
        try:
            detail = resp.data if resp is not None else None
        except Exception:
            detail = None
        return {"ok": False, "operation": op, "error": str(exc), "detail": detail}


if __name__ == "__main__":
    print(json.dumps(invoke("slack_action", {"operation": "auth_test"}), indent=2))
