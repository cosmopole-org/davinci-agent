#!/usr/bin/env python3
"""deploy_davinci_agent.py — deploy ONLY the Davinci agent creature onto a
already-running Caspar node and print its program id.

This is the lean deploy path used by the Decillion e2e script to wire a **real**
Davinci docker agent as the backend's proxy target (as opposed to the offline
``davinciStub``). Unlike ``e2e_test.py`` it does not deploy the tool creatures
or run prompt scenarios — the Decillion backend supplies each agent's tool
catalog from its space at prompt time — so this stays fast and needs no live
LLM round-trip to succeed.

The LLM backbone credentials (``GEMINI_API_KEY`` / ``LLM_PROVIDER`` + its key,
read from the environment) are **baked into the creature image** so the agent
reasons with a real provider even though the backend proxy signals it without an
LLM config. Keys are read from the environment only; never written to disk here.

Connection (plaintext TCP, matching the casparctl local node):
    CASPAR_NODE_HOST   node host   (default 127.0.0.1)
    CASPAR_NODE_PORT   node TCP    (default 8074)
    CASPAR_CA_BUNDLE   CA bundle baked into the image for egress TLS
                       (default /etc/ssl/certs/ca-certificates.crt)

Output (stdout, machine-readable — the caller greps these):
    DAVINCI_PROGRAM_ID=<id>
    DAVINCI_ENTITY_ID=davinci
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deploy_and_test as dt  # noqa: E402  (canonical deploy helpers)
from davinci.caspar_signaling import CasparSignalingClient  # noqa: E402


def main() -> int:
    host = os.environ.get("CASPAR_NODE_HOST", dt.NODE_HOST)
    port = int(os.environ.get("CASPAR_NODE_PORT", str(dt.NODE_PORT)))
    dt.info(f"connecting to Caspar node {host}:{port}")
    c = CasparSignalingClient(host, port, timeout=120).connect()
    c.login(dt.ADMIN_USER)
    dt.ok(f"logged in as {dt.ADMIN_USER} (user_id={c.user_id})")

    bake = dt.llm_bake_env()
    if bake:
        provider = bake.get("LLM_PROVIDER", "gemini")
        dt.info(f"baking LLM backbone into the creature image (provider={provider})")
    else:
        dt.warn("no LLM key in the environment — the agent will fall back to the "
                "heuristic reasoner (set GEMINI_API_KEY for real reasoning)")

    davinci = dt.deploy_davinci(c, bake_env=bake)
    c.close()

    # Machine-readable markers for the caller.
    print("DAVINCI_PROGRAM_ID=" + davinci["program_id"], flush=True)
    print("DAVINCI_ENTITY_ID=" + davinci["entity_id"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
