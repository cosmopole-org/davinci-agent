"""browser_automation tool creature — real headless browser via Playwright.

Drives a headless Chromium through an ordered list of steps and returns a
per-step result log plus the final page title/url. Screenshots are returned as
base64 PNG so they survive the signalling channel.

Signal payload (function ``automate`` or ``invoke``)::

    {
      "url": "https://example.com",      # optional initial navigation
      "browser": "chromium",              # chromium | firefox | webkit
      "headless": true,
      "viewport": {"width": 1280, "height": 800},
      "user_agent": "...",
      "timeout_ms": 15000,                 # per-action default
      "steps": [
        {"action": "goto", "url": "https://example.com"},
        {"action": "wait_for_selector", "selector": "h1"},
        {"action": "text", "selector": "h1"},
        {"action": "fill", "selector": "#q", "value": "davinci"},
        {"action": "click", "selector": "button[type=submit]"},
        {"action": "screenshot", "full_page": true},
        {"action": "evaluate", "script": "document.title"}
      ]
    }

Supported step actions: ``goto``, ``click``, ``dblclick``, ``fill``, ``type``,
``press``, ``hover``, ``check``, ``uncheck``, ``select_option``, ``scroll``,
``wait_for_selector``, ``wait_for_timeout``, ``wait_for_load_state``,
``screenshot``, ``pdf``, ``text``, ``inner_html``, ``get_attribute``,
``content``, ``title``, ``url``, ``set_viewport``, ``evaluate``, ``go_back``,
``reload``.
"""

from __future__ import annotations

import base64
import json
import os


# Map the natural action names LLMs commonly emit onto the canonical Playwright
# step verbs this tool implements, so a model that asks to "navigate" or "wait"
# (instead of "goto" / "wait_for_*") still drives the browser successfully.
_ACTION_ALIASES = {
    "navigate": "goto", "open": "goto", "open_url": "goto", "visit": "goto",
    "load": "goto", "go": "goto", "goto_url": "goto",
    "sleep": "wait_for_timeout", "delay": "wait_for_timeout", "pause": "wait_for_timeout",
    "wait_for": "wait_for_selector", "wait_for_element": "wait_for_selector",
    "wait_for_navigation": "wait_for_load_state", "wait_for_network": "wait_for_load_state",
    "get_text": "text", "extract_text": "text", "read_text": "text", "inner_text": "text",
    "extract": "content", "get_content": "content", "get_html": "content", "html": "content",
    "page_content": "content", "source": "content",
    "scroll_down": "scroll", "scroll_to_bottom": "scroll", "scroll_to": "scroll",
    "eval": "evaluate", "execute_script": "evaluate", "run_script": "evaluate",
    "snap": "screenshot", "capture": "screenshot", "shot": "screenshot",
    "back": "go_back", "attribute": "get_attribute", "attr": "get_attribute",
}


def _resolve_action(step: dict) -> str:
    action = (step.get("action") or step.get("op") or "").lower().strip()
    # ``wait`` is ambiguous: pick the concrete verb from the step's own fields.
    if action == "wait":
        if step.get("selector"):
            return "wait_for_selector"
        if any(k in step for k in ("ms", "timeout", "seconds", "duration")):
            return "wait_for_timeout"
        return "wait_for_load_state"
    return _ACTION_ALIASES.get(action, action)


def _run_step(page, step: dict, default_timeout: int) -> dict:
    action = _resolve_action(step)
    sel = step.get("selector")
    timeout = int(step.get("timeout_ms", default_timeout))
    out = {"action": action}

    if action == "wait_for_timeout" and "ms" not in step:
        # Honour seconds/duration aliases when an LLM expresses the pause that way.
        secs = step.get("seconds", step.get("duration"))
        if secs is not None:
            step = {**step, "ms": int(float(secs) * 1000)}

    if action == "goto":
        resp = page.goto(step["url"], timeout=timeout,
                         wait_until=step.get("wait_until", "load"))
        out.update({"url": page.url, "status": resp.status if resp else None})
    elif action == "click":
        page.click(sel, timeout=timeout)
        out["clicked"] = sel
    elif action == "dblclick":
        page.dblclick(sel, timeout=timeout)
        out["dblclicked"] = sel
    elif action in ("fill", "type"):
        if action == "type":
            page.type(sel, step.get("value", ""), timeout=timeout,
                      delay=int(step.get("delay", 0)))
        else:
            page.fill(sel, step.get("value", ""), timeout=timeout)
        out.update({"selector": sel, "value": step.get("value", "")})
    elif action == "press":
        page.press(sel or "body", step.get("key", "Enter"), timeout=timeout)
        out["key"] = step.get("key", "Enter")
    elif action == "hover":
        page.hover(sel, timeout=timeout)
        out["hovered"] = sel
    elif action in ("check", "uncheck"):
        getattr(page, action)(sel, timeout=timeout)
        out[action] = sel
    elif action == "select_option":
        vals = page.select_option(sel, step.get("value"), timeout=timeout)
        out["selected"] = vals
    elif action == "scroll":
        page.evaluate("([x,y]) => window.scrollTo(x,y)",
                      [int(step.get("x", 0)), int(step.get("y", 10_000))])
        out["scrolled"] = True
    elif action == "wait_for_selector":
        page.wait_for_selector(sel, timeout=timeout, state=step.get("state", "visible"))
        out["found"] = sel
    elif action == "wait_for_timeout":
        page.wait_for_timeout(int(step.get("ms", 1000)))
        out["waited_ms"] = int(step.get("ms", 1000))
    elif action == "wait_for_load_state":
        page.wait_for_load_state(step.get("state", "networkidle"), timeout=timeout)
        out["load_state"] = step.get("state", "networkidle")
    elif action == "screenshot":
        png = page.screenshot(full_page=bool(step.get("full_page", False)))
        out["screenshot_base64"] = base64.b64encode(png).decode()
        out["bytes"] = len(png)
    elif action == "pdf":
        try:
            pdf = page.pdf()
            out["pdf_base64"] = base64.b64encode(pdf).decode()
        except Exception as exc:
            out["error"] = f"pdf only supported in headless chromium: {exc}"
    elif action in ("text", "inner_text"):
        out["text"] = page.inner_text(sel, timeout=timeout) if sel else page.inner_text("body")
    elif action == "inner_html":
        out["html"] = page.inner_html(sel, timeout=timeout)
    elif action == "get_attribute":
        out["value"] = page.get_attribute(sel, step.get("name", "href"), timeout=timeout)
    elif action == "content":
        out["content"] = page.content()[:200_000]
    elif action == "title":
        out["title"] = page.title()
    elif action == "url":
        out["url"] = page.url
    elif action == "set_viewport":
        page.set_viewport_size({"width": int(step.get("width", 1280)),
                                "height": int(step.get("height", 800))})
        out["viewport_set"] = True
    elif action == "evaluate":
        out["result"] = page.evaluate(step.get("script", "1+1"))
    elif action == "go_back":
        page.go_back(timeout=timeout)
        out["url"] = page.url
    elif action == "reload":
        page.reload(timeout=timeout)
        out["url"] = page.url
    else:
        raise ValueError(f"unknown browser action '{action}'")

    out["ok"] = True
    return out


def invoke(function_name: str, payload: dict) -> dict:
    payload = payload or {}
    from playwright.sync_api import sync_playwright

    steps = list(payload.get("steps") or [])
    if payload.get("url"):
        steps.insert(0, {"action": "goto", "url": payload["url"]})
    if not steps:
        return {"ok": False, "error": "no 'steps' or 'url' provided"}

    browser_name = (payload.get("browser") or "chromium").lower()
    headless = bool(payload.get("headless", True))
    default_timeout = int(payload.get("timeout_ms", 15000))
    results = []
    ok_all = True

    # Tool creatures reach the network through a TLS-intercepting egress gateway
    # whose CA is baked into the image as a file bundle (honoured by requests via
    # REQUESTS_CA_BUNDLE). Chromium, however, uses its own trust store and will
    # not pick that bundle up, so it would otherwise fail every HTTPS navigation
    # with net::ERR_CERT_AUTHORITY_INVALID. Default to ignoring cert errors here
    # (overridable per-call or via env) so the browser can load real sites.
    ignore_tls = payload.get("ignore_https_errors")
    if ignore_tls is None:
        ignore_tls = os.environ.get(
            "BROWSER_IGNORE_HTTPS_ERRORS", "1").strip().lower() not in ("0", "false", "no", "")

    try:
        with sync_playwright() as pw:
            launcher = getattr(pw, browser_name)
            chromium_args = ["--no-sandbox", "--disable-dev-shm-usage"]
            if ignore_tls:
                chromium_args.append("--ignore-certificate-errors")
            browser = launcher.launch(headless=headless,
                                      args=chromium_args
                                      if browser_name == "chromium" else None)
            ctx_kwargs = {}
            if payload.get("viewport"):
                ctx_kwargs["viewport"] = payload["viewport"]
            if payload.get("user_agent"):
                ctx_kwargs["user_agent"] = payload["user_agent"]
            if ignore_tls:
                ctx_kwargs["ignore_https_errors"] = True
            context = browser.new_context(**ctx_kwargs)
            context.set_default_timeout(default_timeout)
            page = context.new_page()

            for step in steps:
                resolved = _resolve_action(step)
                try:
                    results.append(_run_step(page, step, default_timeout))
                except Exception as exc:
                    ok_all = False
                    results.append({"action": step.get("action"), "ok": False, "error": str(exc)})
                    # A failed ``wait_for_selector`` (e.g. the caller guessed a
                    # selector that this dynamic site doesn't use) must not abort
                    # the whole run: the later extraction steps — and the rendered
                    # body text attached below — still yield usable content.
                    soft = resolved in ("wait_for_selector", "wait_for_load_state",
                                        "wait_for_timeout")
                    if soft:
                        continue
                    if not step.get("continue_on_error", payload.get("continue_on_error", False)):
                        break

            # Always return the rendered page text + a content excerpt so the
            # caller receives the actual (post-JS) content even when its own
            # selector-based extraction steps missed. This is what lets an agent
            # parse dynamically-loaded posts it couldn't target by selector.
            final = {"url": page.url, "title": page.title()}
            try:
                final["text"] = (page.evaluate("document.body.innerText") or "")[:40_000]
            except Exception:
                pass
            try:
                final["content_excerpt"] = page.content()[:40_000]
            except Exception:
                pass
            context.close()
            browser.close()
    except Exception as exc:
        return {"ok": False, "error": f"browser launch/automation failed: {exc}",
                "steps": results}

    return {"ok": ok_all, "function": function_name, "browser": browser_name,
            "final": final, "step_count": len(results), "steps": results}


if __name__ == "__main__":
    res = invoke("automate", {"url": "https://example.com",
                              "steps": [{"action": "text", "selector": "h1"},
                                        {"action": "title"}]})
    # Trim base64 blobs for readability.
    print(json.dumps(res, indent=2)[:1200])
