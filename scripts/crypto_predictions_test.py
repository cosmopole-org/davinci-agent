#!/usr/bin/env python3
"""End-to-end crypto-predictions test of Davinci on Caspar (signaling paradigm).

Every interaction flows through the Caspar **signaling API** (the docker-host
bridge gateway). Nothing is driven by uploaded task files: tools and the Davinci
agent are each started **once** as standalone, long-lived machine-program VMs via
``runEntity`` and thereafter only communicate by signals.

  Phase 1  Deploy each tool as its own ``docker`` creature (build its image).
  Phase 2  Start each tool as a standalone serving VM (``runEntity``); it
           connects to the gateway and waits for invoke signals. Smoke-signal
           each one over the signaling API and assert its reply.
  Phase 3  Deploy + start the DecillionAI ``stores`` space creature; create a
           space and add the Davinci agent + every tool as members.
  Phase 4  Deploy the Davinci agent creature and start it as a standalone
           serving VM; it connects to the gateway and waits for its task signal.
  Phase 5  Send the crypto-predictions task to Davinci over the signaling API and
           await its result signal. Davinci drives the tool creatures by
           signalling their live VMs — creature-to-creature, no task files.

The Gemini API key is read from $GEMINI_API_KEY and is delivered to Davinci
inside the task signal; it is never written to disk or committed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import deploy_and_test as dt  # noqa: E402
from davinci.caspar_signaling import CasparSignalingClient  # noqa: E402

GREEN, RED, CYAN, YELLOW, NC = dt.GREEN, dt.RED, dt.CYAN, "\033[0;33m", dt.NC


def info(m): print(f"{CYAN}[crypto-e2e]{NC} {m}", flush=True)
def ok(m):   print(f"{GREEN}[ ok ]{NC} {m}", flush=True)
def bad(m):  print(f"{RED}[fail]{NC} {m}", flush=True)
def warn(m): print(f"{YELLOW}[warn]{NC} {m}", flush=True)


TASK_OBJECTIVE = (
    "fetch and check this coin market cap trader profile: "
    "https://coinmarketcap.com/community/profile/vlad_anderson/\n"
    "And extract its 10 latest posts about crypto prices prediction "
    "(including analysing the screenshots he attached of charts attached to "
    "the posts) and then compare the predicted prices they talk about to the "
    "equivalent real happened price of those cryptos at the specified "
    "predicting time points by fetching the real crypto prices using those "
    "time and date that had been being predicted, give each of his "
    "predictions a score from 0 to 100 based on the accuracy of prediction "
    "compared to real prices that happened and finally give me only the "
    "average prediction score number of all those predictions. "
    "The output must only be a number."
)

TOOL_IDS = ["web_search", "fetch_url", "browser_automation", "python_exec"]

# Long enough that every tool VM stays alive serving signals for the whole
# duration of the Davinci run; the tool also self-exits after this idle window.
TOOL_SERVE_SECONDS = 1200
DAVINCI_SERVE_SECONDS = 1200

STORES_WASM = os.environ.get(
    "STORES_WASM",
    str((REPO.parent / "decillionai-server" / "wasm" / "stores.wasm")))


# --------------------------------------------------------------------------- #
# Phase 2 — start tools as standalone serving VMs + smoke-signal each
# --------------------------------------------------------------------------- #

def start_tool_serving(c: CasparSignalingClient, rec: dict) -> bool:
    vm_id = c.run_entity(rec["program_id"], rec["entity_id"], params={},
                         ram_mb=320, max_exec_seconds=TOOL_SERVE_SECONDS)
    rec["serve_vm"] = vm_id
    found, _ = c.wait_for_vm_log(vm_id, "TOOL_SERVE_READY", timeout=150, poll=3)
    (ok if found else bad)(
        f"tool {rec['tool_id']} {'serving' if found else 'FAILED to serve'} (vm={vm_id})")
    return found


def smoke_signal_tool(c: CasparSignalingClient, rec: dict) -> bool:
    envelope = {"kind": "invoke", "tool_id": rec["tool_id"],
                "function": rec.get("function", "invoke"),
                "payload": {"task": f"smoke {rec['tool_id']}", "query": "davinci caspar",
                            "url": "https://example.com", "ignore_https_errors": True,
                            "code": "result = 6*7", "sql": "SELECT 1 AS one",
                            "documents": ["davinci orchestrates caspar creatures"]}}
    resp = c.signal_entity_await(creature_id=rec["machine_id"], program_id=rec["program_id"],
                                 entity_id=rec["entity_id"], envelope=envelope, timeout=150)
    reply = resp.get("result") if isinstance(resp, dict) else None
    good = bool(resp.get("ok") and isinstance(reply, dict)
                and reply.get("kind") == "tools/result")
    if good:
        ok(f"signalled {rec['tool_id']} → {json.dumps(reply.get('result', {}))[:110]}")
    else:
        bad(f"no signal reply from {rec['tool_id']}: {str(resp)[:160]}")
    return good


# --------------------------------------------------------------------------- #
# Phase 3 — space (store) + members
# --------------------------------------------------------------------------- #

def setup_store_with_members(c: CasparSignalingClient, tool_recs: list, davinci: dict) -> dict:
    wasm = Path(STORES_WASM)
    if not wasm.exists():
        warn(f"stores.wasm not found at {wasm} — skipping space (membership unavailable)")
        return {}
    stores = dt.deploy_wasm_creature(c, "stores", wasm, entity_id="main")
    target = {"creature_id": stores["machine_id"],
              "program_id": stores["program_id"], "entity": "main"}

    resp = c.signal_miniapp(
        creature_id=target["creature_id"], program_id=target["program_id"],
        entity="main", action="create",
        payload={"isPublic": True, "persHist": False, "origin": "davinci-crypto-space",
                 "metadata": {"name": "davinci-crypto-space"}})
    store_id = ""
    if resp.get("ok"):
        res = resp.get("result", {})
        store_id = res.get("storeId") or (res.get("host") or {}).get("storeId") or ""
    if not store_id:
        warn(f"could not create space: {resp}")
        return {}
    ok(f"created space (store) {store_id}")

    added = 0
    for t in tool_recs:
        meta = dict(dt.tool_full_metadata(t["tool_id"]))
        meta["program_id"] = t["program_id"]; meta["entity_id"] = t["entity_id"]
        r = c.signal_miniapp(
            creature_id=target["creature_id"], program_id=target["program_id"],
            entity="main", action="addMachine", store_id=store_id,
            payload={"storeId": store_id, "programId": t["program_id"],
                     "machineId": t["machine_id"], "metadata": meta})
        if r.get("ok"):
            added += 1
            ok(f"added tool {t['tool_id']} to space {store_id}")
        else:
            warn(f"addMachine {t['tool_id']} failed: {r.get('error')}")

    davinci_meta = {"tool_id": "davinci_agent", "name": "davinci",
                    "description": "Davinci orchestration agent",
                    "categories": ["agent", "orchestrator"], "risk_level": "medium",
                    "program_id": davinci["program_id"], "entity_id": davinci["entity_id"]}
    r = c.signal_miniapp(
        creature_id=target["creature_id"], program_id=target["program_id"],
        entity="main", action="addMachine", store_id=store_id,
        payload={"storeId": store_id, "programId": davinci["program_id"],
                 "machineId": davinci["machine_id"], "metadata": davinci_meta})
    (ok if r.get("ok") else warn)(
        f"added davinci agent to space {store_id} as member" if r.get("ok")
        else f"addMachine davinci failed: {r.get('error')}")
    ok(f"added {added}/{len(tool_recs)} tools (+ davinci) to space {store_id}")
    return {**target, "store_id": store_id}


# --------------------------------------------------------------------------- #
# Phase 4/5 — start Davinci serving, then signal the task
# --------------------------------------------------------------------------- #

def start_davinci_serving(c: CasparSignalingClient, davinci: dict) -> bool:
    vm_id = c.run_entity(davinci["program_id"], "davinci", params={},
                         ram_mb=768, max_exec_seconds=DAVINCI_SERVE_SECONDS)
    davinci["serve_vm"] = vm_id
    found, _ = c.wait_for_vm_log(vm_id, "DAVINCI_READY", timeout=200, poll=3)
    (ok if found else bad)(
        f"davinci {'serving (awaiting task signal)' if found else 'FAILED to reach READY'} (vm={vm_id})")
    return found


def signal_davinci_task(c: CasparSignalingClient, davinci: dict, tool_recs: list,
                        store_cfg: dict) -> dict:
    config = {
        "node_host": dt.NODE_HOST_FROM_VM, "node_port": dt.NODE_PORT,
        "username": dt.ADMIN_USER,
        "gemini_api_key": dt.GEMINI_API_KEY,
        "gemini_models": dt.GEMINI_MODELS or None,
        "tools": [
            {"name": t["name"], "tool_id": t["tool_id"], "category": t["category"],
             "risk": t["risk"], "description": t["description"],
             "program_id": t["program_id"], "entity_id": t["entity_id"],
             "machine_id": t["machine_id"], "function": t.get("function", "invoke"),
             "requires_network": t.get("requires_network", False)}
            for t in tool_recs],
    }
    if store_cfg:
        config["store"] = store_cfg
    envelope = {"kind": "task", "objective": TASK_OBJECTIVE,
                "required_categories": list({t["category"] for t in tool_recs}),
                "config": config}
    info("signalling Davinci with the crypto-predictions task (over signaling API)…")
    resp = c.signal_entity_await(creature_id=davinci["machine_id"], program_id=davinci["program_id"],
                                 entity_id="davinci", envelope=envelope,
                                 timeout=DAVINCI_SERVE_SECONDS)
    return resp


def main() -> int:
    if not dt.GEMINI_API_KEY:
        bad("GEMINI_API_KEY not set — exiting (Gemini is the LLM backbone)")
        return 2

    tools = [t for t in dt.TOOLS if t["tool_id"] in TOOL_IDS]
    info(f"selected tools: {[t['tool_id'] for t in tools]}")
    info(f"connecting to Caspar node {dt.NODE_HOST}:{dt.NODE_PORT}")
    c = CasparSignalingClient(dt.NODE_HOST, dt.NODE_PORT, timeout=120).connect()
    c.login(dt.ADMIN_USER)
    ok(f"logged in as {dt.ADMIN_USER} (user_id={c.user_id})")

    summary = {"tools_selected": [t["tool_id"] for t in tools], "tools_deployed": 0,
               "tools_serving": 0, "tools_signalled": 0, "davinci_serving": False,
               "store_id": "", "result": None}

    info("── Phase 1: deploy tool creatures (build images) ──")
    tool_recs = []
    for tool in tools:
        rec = dt.deploy_tool(c, tool)
        tool_recs.append(rec)
        summary["tools_deployed"] += 1

    info("── Phase 2: start tools as standalone serving VMs + smoke-signal ──")
    for rec in tool_recs:
        if start_tool_serving(c, rec):
            summary["tools_serving"] += 1
            if smoke_signal_tool(c, rec):
                summary["tools_signalled"] += 1

    info("── Phase 3: deploy davinci creature (build image) ──")
    davinci = dt.deploy_davinci(c)

    info("── Phase 4: deploy stores space creature + create space + add members ──")
    store_cfg = setup_store_with_members(c, tool_recs, davinci)
    summary["store_id"] = store_cfg.get("store_id", "")

    info("── Phase 5: start Davinci serving, then signal the task ──")
    davinci_ready = start_davinci_serving(c, davinci)
    summary["davinci_serving"] = davinci_ready
    result_dict = {}
    if davinci_ready:
        resp = signal_davinci_task(c, davinci, tool_recs, store_cfg)
        reply = resp.get("result") if isinstance(resp, dict) else None
        if resp.get("ok") and isinstance(reply, dict):
            result_dict = reply.get("result", {}) or {}
            ok("received Davinci result signal")
        else:
            bad(f"no Davinci result signal: {str(resp)[:200]}")
        summary["result"] = result_dict

    c.close()
    print("\n" + "=" * 64)
    print("CRYPTO_E2E_RESULT " + json.dumps(summary, indent=2, default=str))
    print("=" * 64)
    success = bool(summary["tools_serving"] == len(tools)
                   and summary["tools_signalled"] == len(tools)
                   and summary["davinci_serving"] and result_dict)
    (ok if success else bad)("OVERALL: " + ("PASS" if success else "FAIL"))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
