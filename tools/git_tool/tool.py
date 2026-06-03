"""git_tool creature — real git operations + GitHub pull-request creation.

Two routed functions (per the tool metadata):

``status_commit`` — inspect/stage/commit a repository::

    {
      "repo_path": "/app/workspace/repo",     # working tree (created if needed)
      "repo_url": "https://github.com/o/r",   # clone here if repo_path missing
      "action": "auto",                        # auto|status|add|commit|diff|log|
                                               # branch|checkout|init|push|pull
      "message": "commit message",             # for commit (triggers add+commit)
      "files": ["a.py"],                        # paths to stage (default: all)
      "branch": "feature/x",                   # for checkout/branch/push
      "author_name": "...", "author_email": "...",
      "remote": "origin"
    }

``open_pr`` — push the branch and open/locate a GitHub PR::

    {
      "repo_path": "...", "repo": "owner/name",
      "title": "...", "body": "...",
      "head": "feature/x", "base": "main",
      "push": true, "remote": "origin",
      "github_token": "..."                     # or GITHUB_TOKEN / GH_TOKEN env
    }

Authentication for clone/push uses ``GITHUB_TOKEN``/``GH_TOKEN`` (or
``payload['github_token']``) injected into ``https://`` remotes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from urllib.parse import urlparse

DEFAULT_REPO = "/app/workspace/repo"


def _token(payload: dict) -> str:
    return (payload.get("github_token") or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN") or "")


def _auth_url(url: str, token: str) -> str:
    if token and url.startswith("https://") and "@" not in urlparse(url).netloc:
        return url.replace("https://", f"https://x-access-token:{token}@", 1)
    return url


def _git(args, cwd=None, token: str = "") -> dict:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_AUTHOR_NAME", env.get("GIT_AUTHOR_NAME", "Davinci Agent"))
    env.setdefault("GIT_AUTHOR_EMAIL", env.get("GIT_AUTHOR_EMAIL", "davinci@cosmopole.org"))
    env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
    env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          env=env, timeout=180)
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if token:  # never echo a token back in command output
        out = out.replace(token, "***")
        err = err.replace(token, "***")
    return {"cmd": "git " + " ".join(args), "rc": proc.returncode, "stdout": out, "stderr": err}


def _ensure_repo(payload: dict, token: str) -> tuple:
    repo_path = payload.get("repo_path") or DEFAULT_REPO
    steps = []
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        repo_url = payload.get("repo_url")
        os.makedirs(os.path.dirname(repo_path) or ".", exist_ok=True)
        if repo_url:
            steps.append(_git(["clone", _auth_url(repo_url, token), repo_path], token=token))
        else:
            os.makedirs(repo_path, exist_ok=True)
            steps.append(_git(["init", repo_path], token=token))
    # gVisor/foreign-owned trees: mark as safe so git won't refuse to operate.
    _git(["config", "--global", "--add", "safe.directory", repo_path], token=token)
    return repo_path, steps


def _current_branch(repo_path: str) -> str:
    r = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
    return r["stdout"] or "HEAD"


def _status_commit(payload: dict) -> dict:
    token = _token(payload)
    repo_path, steps = _ensure_repo(payload, token)
    action = (payload.get("action") or "auto").lower()
    remote = payload.get("remote", "origin")

    if payload.get("author_name"):
        _git(["config", "user.name", payload["author_name"]], cwd=repo_path)
    if payload.get("author_email"):
        _git(["config", "user.email", payload["author_email"]], cwd=repo_path)

    def run(*a):
        r = _git(list(a), cwd=repo_path, token=token)
        steps.append(r)
        return r

    if action in ("branch", "checkout") and payload.get("branch"):
        run("checkout", "-B", payload["branch"])
    elif action == "branch":
        run("branch", "--show-current")

    if action == "diff":
        run("--no-pager", "diff", *( ["--cached"] if payload.get("cached") else [] ))
    if action == "log":
        run("--no-pager", "log", "--oneline", "-n", str(payload.get("n", 10)))
    if action == "pull":
        run("pull", remote, payload.get("branch", _current_branch(repo_path)))

    do_commit = action in ("commit", "auto") and payload.get("message")
    if action in ("add", "commit", "auto"):
        files = payload.get("files")
        run("add", *(files if files else ["-A"]))
    if do_commit:
        run("commit", "--no-gpg-sign", "-m", str(payload["message"]))

    if action == "push" or payload.get("push"):
        branch = payload.get("branch") or _current_branch(repo_path)
        if payload.get("repo_url"):
            run("remote", "set-url", remote, _auth_url(payload["repo_url"], token))
        run("push", "-u", remote, branch)

    status = _git(["--no-pager", "status", "--porcelain=v1", "-b"], cwd=repo_path, token=token)
    head = _git(["--no-pager", "log", "-1", "--pretty=%H %s"], cwd=repo_path)
    ok = all(s["rc"] == 0 for s in steps) and status["rc"] == 0
    return {"ok": ok, "function": "status_commit", "repo_path": repo_path,
            "branch": _current_branch(repo_path),
            "head": head["stdout"], "status": status["stdout"], "steps": steps}


def _gh_repo_slug(payload: dict, repo_path: str, token: str) -> str:
    slug = payload.get("repo")
    if slug:
        return slug
    r = _git(["remote", "get-url", payload.get("remote", "origin")], cwd=repo_path, token=token)
    m = re.search(r"github\.com[:/]+([^/]+/[^/.]+)", r["stdout"])
    return m.group(1) if m else ""


def _open_pr(payload: dict) -> dict:
    import requests
    token = _token(payload)
    repo_path = payload.get("repo_path") or DEFAULT_REPO
    remote = payload.get("remote", "origin")
    head = payload.get("head") or payload.get("branch") or _current_branch(repo_path)
    base = payload.get("base", "main")
    steps = []

    if payload.get("push", True):
        if payload.get("repo_url"):
            steps.append(_git(["remote", "set-url", remote, _auth_url(payload["repo_url"], token)],
                              cwd=repo_path, token=token))
        steps.append(_git(["push", "-u", remote, head], cwd=repo_path, token=token))

    slug = _gh_repo_slug(payload, repo_path, token)
    if not slug:
        return {"ok": False, "error": "could not determine GitHub repo (set payload.repo='owner/name')",
                "steps": steps}
    if not token:
        return {"ok": False, "error": "no GitHub token (set GITHUB_TOKEN or payload.github_token)",
                "repo": slug, "steps": steps}

    api = f"https://api.github.com/repos/{slug}/pulls"
    hdrs = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}
    body = {"title": payload.get("title", f"Davinci: {head}"), "head": head,
            "base": base, "body": payload.get("body", "")}
    if payload.get("draft"):
        body["draft"] = True
    try:
        resp = requests.post(api, headers=hdrs, json=body, timeout=30)
        if resp.status_code == 201:
            pr = resp.json()
            return {"ok": True, "function": "open_pr", "repo": slug, "action": "created",
                    "number": pr["number"], "url": pr["html_url"], "head": head, "base": base,
                    "steps": steps}
        # Already exists? Locate it.
        if resp.status_code == 422:
            existing = requests.get(api, headers=hdrs, timeout=30,
                                    params={"head": f"{slug.split('/')[0]}:{head}", "state": "open"})
            items = existing.json() if existing.ok else []
            if items:
                pr = items[0]
                return {"ok": True, "function": "open_pr", "repo": slug, "action": "exists",
                        "number": pr["number"], "url": pr["html_url"], "head": head, "base": base,
                        "steps": steps}
        return {"ok": False, "function": "open_pr", "repo": slug, "status": resp.status_code,
                "error": resp.json() if resp.content else resp.reason, "steps": steps}
    except Exception as exc:
        return {"ok": False, "function": "open_pr", "repo": slug, "error": str(exc), "steps": steps}


def invoke(function_name: str, payload: dict) -> dict:
    payload = payload or {}
    fn = (function_name or "").lower()
    if fn == "open_pr" or payload.get("action") == "open_pr":
        return _open_pr(payload)
    return _status_commit(payload)


if __name__ == "__main__":
    import tempfile
    repo = tempfile.mkdtemp(prefix="gitdemo_")
    with open(os.path.join(repo, "hello.txt"), "w") as fh:
        fh.write("davinci\n")
    print(json.dumps(invoke("status_commit",
          {"repo_path": repo, "message": "initial commit"}), indent=2))
