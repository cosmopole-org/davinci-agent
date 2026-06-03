"""fetch_url tool creature — real HTTP fetch with content extraction.

Performs an HTTP request with ``requests`` and, for HTML responses, extracts a
clean title / readable text / links with BeautifulSoup. JSON responses are
parsed; binary responses are summarised (and optionally base64-returned).

Signal payload (function ``fetch`` or ``invoke``)::

    {
      "url": "https://example.com",   # required
      "method": "GET",                # GET/POST/PUT/PATCH/DELETE/HEAD
      "headers": {...},
      "params": {...},                 # query string
      "data": "..." | {...},          # form/raw body
      "json": {...},                   # JSON body (sets content-type)
      "timeout": 20,
      "max_bytes": 2000000,
      "extract": true,                 # extract readable text + links (HTML)
      "render_binary_base64": false
    }
"""

from __future__ import annotations

import base64
import json
from urllib.parse import urljoin, urlparse

import requests

DEFAULT_TIMEOUT = 20
DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_UA = "Mozilla/5.0 (compatible; DavinciFetch/1.0; +https://github.com/cosmopole-org)"


def _extract_html(html: str, base_url: str) -> dict:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml") if _has_lxml() else BeautifulSoup(html, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else None
    meta_desc = None
    tag = soup.find("meta", attrs={"name": "description"}) or \
        soup.find("meta", attrs={"property": "og:description"})
    if tag and tag.get("content"):
        meta_desc = tag["content"].strip()

    for bad in soup(["script", "style", "noscript", "template", "svg"]):
        bad.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())

    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if href in seen or href.startswith("javascript:"):
            continue
        seen.add(href)
        links.append({"text": " ".join(a.get_text(" ", strip=True).split())[:120], "href": href})
        if len(links) >= 100:
            break

    return {"title": title, "description": meta_desc, "text": text[:50_000],
            "text_length": len(text), "links": links}


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401
        return True
    except Exception:
        return False


def invoke(function_name: str, payload: dict) -> dict:
    payload = payload or {}
    url = payload.get("url") or payload.get("task")
    if not url:
        return {"ok": False, "error": "no 'url' provided"}
    if not urlparse(str(url)).scheme:
        url = "https://" + str(url)

    method = str(payload.get("method", "GET")).upper()
    timeout = max(1, min(int(payload.get("timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT), 120))
    max_bytes = int(payload.get("max_bytes", DEFAULT_MAX_BYTES) or DEFAULT_MAX_BYTES)
    headers = {"User-Agent": DEFAULT_UA}
    headers.update(payload.get("headers") or {})

    try:
        resp = requests.request(
            method, url, headers=headers, params=payload.get("params"),
            data=payload.get("data"), json=payload.get("json"),
            timeout=timeout, allow_redirects=payload.get("allow_redirects", True),
            stream=True,
        )
        content = resp.raw.read(max_bytes + 1, decode_content=True)
        truncated = len(content) > max_bytes
        content = content[:max_bytes]
    except Exception as exc:
        return {"ok": False, "url": url, "error": str(exc)}

    ctype = resp.headers.get("Content-Type", "").lower()
    out = {
        "ok": resp.ok,
        "url": resp.url,
        "status": resp.status_code,
        "method": method,
        "content_type": ctype,
        "headers": dict(resp.headers),
        "bytes": len(content),
        "truncated": truncated,
        "function": function_name,
        "elapsed_ms": int(resp.elapsed.total_seconds() * 1000),
    }

    is_text = ctype.startswith("text/") or "json" in ctype or "xml" in ctype or "html" in ctype
    if "json" in ctype:
        try:
            out["json"] = json.loads(content.decode(resp.encoding or "utf-8", "replace"))
        except Exception:
            is_text = True
    if "json" not in out and is_text:
        body = content.decode(resp.encoding or "utf-8", "replace")
        if "html" in ctype and payload.get("extract", True):
            try:
                out["content"] = _extract_html(body, resp.url)
            except Exception as exc:
                out["extract_error"] = str(exc)
                out["text"] = body[:50_000]
        else:
            out["text"] = body[:50_000]
    elif "json" not in out:
        # Binary payload.
        out["binary"] = True
        if payload.get("render_binary_base64"):
            out["base64"] = base64.b64encode(content).decode()

    return out


if __name__ == "__main__":
    print(json.dumps(invoke("fetch", {"url": "https://example.com"}), indent=2)[:1500])
