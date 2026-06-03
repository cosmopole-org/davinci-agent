"""jira_connector tool creature — real Jira Cloud REST API (v3) operations.

Authentication (Basic auth with an API token):
    JIRA_URL        e.g. https://your-domain.atlassian.net
    JIRA_EMAIL      account email
    JIRA_API_TOKEN  https://id.atlassian.com/manage-profile/security/api-tokens

(or pass ``base_url`` / ``email`` / ``api_token`` in the payload.)

Signal payload (function ``jira_action`` or ``invoke``)::

    {"operation": "get_issue", "issue_key": "PROJ-1", ...}

Operations:
    get_issue        {issue_key, fields?}
    create_issue     {project_key, summary, issue_type?, description?, fields?}
    update_issue     {issue_key, fields|summary|description}
    add_comment      {issue_key, body}
    search           {jql, max_results?, fields?}
    list_transitions {issue_key}
    transition       {issue_key, transition (id or name)}
    assign           {issue_key, account_id}
    delete_issue     {issue_key}
    myself           {}
"""

from __future__ import annotations

import json
import os

import requests

TIMEOUT = 30


def _conf(payload: dict):
    base = (payload.get("base_url") or os.environ.get("JIRA_URL") or "").rstrip("/")
    email = payload.get("email") or os.environ.get("JIRA_EMAIL")
    token = payload.get("api_token") or os.environ.get("JIRA_API_TOKEN")
    if not (base and email and token):
        raise RuntimeError("missing Jira config (JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN)")
    return base, (email, token)


def _adf(text: str) -> dict:
    """Wrap plain text in Atlassian Document Format (required by v3)."""
    return {"type": "doc", "version": 1, "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": text}]}]}


def _req(method, base, path, auth, **kw):
    r = requests.request(method, f"{base}/rest/api/3/{path}", auth=auth,
                         headers={"Accept": "application/json", "Content-Type": "application/json"},
                         timeout=TIMEOUT, **kw)
    if not r.ok:
        detail = r.json() if r.content and "json" in r.headers.get("Content-Type", "") else r.text
        raise RuntimeError(f"Jira API {r.status_code}: {detail}")
    return r.json() if r.content and "json" in r.headers.get("Content-Type", "") else {}


def _op(base, auth, op: str, p: dict) -> dict:
    if op == "myself":
        return _req("GET", base, "myself", auth)

    if op == "get_issue":
        params = {"fields": ",".join(p["fields"])} if p.get("fields") else {}
        return _req("GET", base, f"issue/{p['issue_key']}", auth, params=params)

    if op == "create_issue":
        fields = dict(p.get("fields") or {})
        fields.setdefault("project", {"key": p["project_key"]})
        fields.setdefault("summary", p.get("summary", "Untitled"))
        fields.setdefault("issuetype", {"name": p.get("issue_type", "Task")})
        if p.get("description") and "description" not in fields:
            fields["description"] = _adf(p["description"])
        return _req("POST", base, "issue", auth, json={"fields": fields})

    if op == "update_issue":
        fields = dict(p.get("fields") or {})
        if p.get("summary"):
            fields["summary"] = p["summary"]
        if p.get("description"):
            fields["description"] = _adf(p["description"])
        _req("PUT", base, f"issue/{p['issue_key']}", auth, json={"fields": fields})
        return {"updated": p["issue_key"], "fields": list(fields)}

    if op == "add_comment":
        return _req("POST", base, f"issue/{p['issue_key']}/comment", auth,
                    json={"body": _adf(p.get("body", ""))})

    if op == "search":
        body = {"jql": p.get("jql", "order by created DESC"),
                "maxResults": int(p.get("max_results", 25))}
        if p.get("fields"):
            body["fields"] = p["fields"]
        data = _req("POST", base, "search", auth, json=body)
        return {"total": data.get("total"),
                "issues": [{"key": i["key"], "summary": i.get("fields", {}).get("summary"),
                            "status": (i.get("fields", {}).get("status") or {}).get("name")}
                           for i in data.get("issues", [])]}

    if op == "list_transitions":
        return _req("GET", base, f"issue/{p['issue_key']}/transitions", auth)

    if op == "transition":
        tr = str(p["transition"])
        if not tr.isdigit():
            avail = _req("GET", base, f"issue/{p['issue_key']}/transitions", auth)
            match = next((t for t in avail.get("transitions", [])
                          if t["name"].lower() == tr.lower()), None)
            if not match:
                raise RuntimeError(f"no transition named '{tr}'")
            tr = match["id"]
        _req("POST", base, f"issue/{p['issue_key']}/transitions", auth,
             json={"transition": {"id": tr}})
        return {"transitioned": p["issue_key"], "transition_id": tr}

    if op == "assign":
        _req("PUT", base, f"issue/{p['issue_key']}/assignee", auth,
             json={"accountId": p.get("account_id")})
        return {"assigned": p["issue_key"], "account_id": p.get("account_id")}

    if op == "delete_issue":
        _req("DELETE", base, f"issue/{p['issue_key']}", auth)
        return {"deleted": p["issue_key"]}

    raise ValueError(f"unknown jira operation '{op}'")


def invoke(function_name: str, payload: dict) -> dict:
    payload = payload or {}
    op = payload.get("operation") or payload.get("action") or "myself"
    try:
        base, auth = _conf(payload)
        data = _op(base, auth, op, payload)
        return {"ok": True, "operation": op, "function": function_name, "result": data}
    except Exception as exc:
        return {"ok": False, "operation": op, "error": str(exc)}


if __name__ == "__main__":
    print(json.dumps(invoke("jira_action", {"operation": "myself"}), indent=2))
