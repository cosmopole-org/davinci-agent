"""web_search tool creature — real web search with pluggable backends.

Backend selection (first available wins, or force with ``payload['engine']``):

* ``tavily``   — TAVILY_API_KEY            (AI search API, returns answer+sources)
* ``brave``    — BRAVE_API_KEY             (Brave Web Search API)
* ``serpapi``  — SERPAPI_API_KEY           (Google results via SerpApi)
* ``google``   — GOOGLE_API_KEY + GOOGLE_CSE_ID  (Programmable Search Engine)
* ``duckduckgo`` — no key required (default fallback, via the ``ddgs`` package)

Signal payload (function ``search`` or ``invoke``)::

    {"query": "...", "max_results": 5, "engine": "duckduckgo", "region": "us-en"}

A bare ``task`` is accepted as the query. Returns a normalised result list of
``{title, url, snippet}`` plus the backend that served it.
"""

from __future__ import annotations

import json
import os

import requests

TIMEOUT = 20


def _query(payload: dict) -> str:
    return str(payload.get("query") or payload.get("q") or payload.get("task") or "").strip()


def _tavily(query: str, n: int, key: str) -> dict:
    r = requests.post("https://api.tavily.com/search", timeout=TIMEOUT, json={
        "api_key": key, "query": query, "max_results": n,
        "include_answer": True, "search_depth": "advanced"})
    r.raise_for_status()
    data = r.json()
    results = [{"title": it.get("title"), "url": it.get("url"),
                "snippet": it.get("content"), "score": it.get("score")}
               for it in data.get("results", [])[:n]]
    return {"engine": "tavily", "answer": data.get("answer"), "results": results}


def _brave(query: str, n: int, key: str) -> dict:
    r = requests.get("https://api.search.brave.com/res/v1/web/search", timeout=TIMEOUT,
                     headers={"X-Subscription-Token": key, "Accept": "application/json"},
                     params={"q": query, "count": n})
    r.raise_for_status()
    web = (r.json().get("web") or {}).get("results", [])
    results = [{"title": it.get("title"), "url": it.get("url"),
                "snippet": it.get("description")} for it in web[:n]]
    return {"engine": "brave", "results": results}


def _serpapi(query: str, n: int, key: str) -> dict:
    r = requests.get("https://serpapi.com/search", timeout=TIMEOUT, params={
        "engine": "google", "q": query, "num": n, "api_key": key})
    r.raise_for_status()
    organic = r.json().get("organic_results", [])
    results = [{"title": it.get("title"), "url": it.get("link"),
                "snippet": it.get("snippet")} for it in organic[:n]]
    return {"engine": "serpapi", "results": results}


def _google_cse(query: str, n: int, key: str, cse_id: str) -> dict:
    r = requests.get("https://www.googleapis.com/customsearch/v1", timeout=TIMEOUT, params={
        "key": key, "cx": cse_id, "q": query, "num": min(n, 10)})
    r.raise_for_status()
    items = r.json().get("items", [])
    results = [{"title": it.get("title"), "url": it.get("link"),
                "snippet": it.get("snippet")} for it in items[:n]]
    return {"engine": "google", "results": results}


def _duckduckgo(query: str, n: int, region: str) -> dict:
    # ddgs (maintained successor of duckduckgo_search); no API key required.
    try:
        from ddgs import DDGS  # type: ignore
    except Exception:
        from duckduckgo_search import DDGS  # type: ignore
    results = []
    with DDGS() as ddgs:
        for it in ddgs.text(query, region=region or "wt-wt", max_results=n):
            results.append({"title": it.get("title"), "url": it.get("href") or it.get("url"),
                            "snippet": it.get("body")})
            if len(results) >= n:
                break
    return {"engine": "duckduckgo", "results": results}


def _pick_engine(payload: dict) -> str:
    forced = (payload.get("engine") or "").lower().strip()
    if forced:
        return forced
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily"
    if os.environ.get("BRAVE_API_KEY"):
        return "brave"
    if os.environ.get("SERPAPI_API_KEY"):
        return "serpapi"
    if os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_CSE_ID"):
        return "google"
    return "duckduckgo"


def invoke(function_name: str, payload: dict) -> dict:
    payload = payload or {}
    query = _query(payload)
    if not query:
        return {"ok": False, "error": "no 'query' provided"}
    n = max(1, min(int(payload.get("max_results", 5) or 5), 25))
    region = payload.get("region", "")
    engine = _pick_engine(payload)

    try:
        if engine == "tavily":
            data = _tavily(query, n, os.environ["TAVILY_API_KEY"])
        elif engine == "brave":
            data = _brave(query, n, os.environ["BRAVE_API_KEY"])
        elif engine == "serpapi":
            data = _serpapi(query, n, os.environ["SERPAPI_API_KEY"])
        elif engine == "google":
            data = _google_cse(query, n, os.environ["GOOGLE_API_KEY"], os.environ["GOOGLE_CSE_ID"])
        else:
            data = _duckduckgo(query, n, region)
    except KeyError as exc:
        return {"ok": False, "error": f"missing credential for engine {engine}: {exc}"}
    except Exception as exc:
        # Fall back to DuckDuckGo if a keyed engine fails and wasn't forced.
        if engine != "duckduckgo" and not payload.get("engine"):
            try:
                data = _duckduckgo(query, n, region)
                data["fallback_from"] = engine
                data["fallback_reason"] = str(exc)
            except Exception as exc2:
                return {"ok": False, "engine": engine, "error": f"{exc}; fallback failed: {exc2}"}
        else:
            return {"ok": False, "engine": engine, "error": str(exc)}

    data.update({"ok": True, "query": query, "count": len(data.get("results", [])),
                 "function": function_name})
    return data


if __name__ == "__main__":
    print(json.dumps(invoke("search", {"query": "anthropic claude", "max_results": 3}), indent=2))
