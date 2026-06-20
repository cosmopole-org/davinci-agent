#!/usr/bin/env python3
"""Full end-to-end test orchestrator for Davinci on Caspar.

This is the Python half of the reproducible workflow (driven by
``scripts/e2e_workflow.sh``). It assumes a Caspar node is already running and
then exercises the whole stack over the signalling / action API:

  Phase 1  Create a machine creature + ``docker`` program for each Davinci tool
           and deploy it (the node builds the image).
  Phase 2  Signal each tool creature directly and assert its ``TOOL_RESPONSE``.
  Phase 3  Create + deploy the Davinci agent itself as a ``docker`` creature.
  Phase 4  Run several **diverse** scenarios: signal the Davinci creature with
           different objectives / required capabilities and assert it (a) boots
           with the Gemini LLM backbone and (b) drives the tool creatures
           creature-to-creature, producing a ``DAVINCI_RESULT``.

The Gemini API key is read from the ``GEMINI_API_KEY`` environment variable
only — it is passed through to the Davinci creature's ``config.json`` and is
never written to disk or committed.

Run (normally via the workflow script):  python3 scripts/e2e_test.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import deploy_and_test as dt  # noqa: E402  (canonical deploy helpers)
from davinci.caspar_signaling import CasparSignalingClient, _log_text  # noqa: E402

GREEN, RED, CYAN, YELLOW, NC = dt.GREEN, dt.RED, dt.CYAN, "\033[0;33m", dt.NC


def info(m): print(f"{CYAN}[e2e]{NC} {m}", flush=True)
def ok(m):   print(f"{GREEN}[ ok ]{NC} {m}", flush=True)
def bad(m):  print(f"{RED}[fail]{NC} {m}", flush=True)
def warn(m): print(f"{YELLOW}[warn]{NC} {m}", flush=True)


# --------------------------------------------------------------------------- #
# Tool selection
# --------------------------------------------------------------------------- #

def _select_tools() -> list:
    """Pick which tool creatures to deploy.

    ``E2E_TOOLS=all`` deploys every tool; otherwise a comma list selects by
    tool_id. The default is a fast, diverse subset that still spans the main
    capability families (research, computation, retrieval, analytics, VCS).
    """
    sel = os.environ.get("E2E_TOOLS", "").strip()
    if sel.lower() == "all" or not sel:
        default = {"python_exec", "web_search", "fetch_url", "vector_search",
                   "sql_query", "git_tool"}
        chosen = None if sel.lower() == "all" else default
    else:
        chosen = {s.strip() for s in sel.split(",") if s.strip()}
    if chosen is None:
        return list(dt.TOOLS)
    return [t for t in dt.TOOLS if t["tool_id"] in chosen]


# --------------------------------------------------------------------------- #
# Diverse Davinci scenarios (Phase 4)
# --------------------------------------------------------------------------- #

SCENARIOS = [
    {
        "name": "research_and_compute",
        "objective": "Research the Caspar protocol for current facts, then compute "
                     "a small numeric summary of the findings.",
        "categories": ["research", "computation"],
    },
    {
        "name": "retrieve_and_analyze",
        "objective": "Retrieve relevant context from the knowledge base and run an "
                     "analytical query to summarise it.",
        "categories": ["knowledge_retrieval", "analytics"],
    },
    {
        "name": "engineering_workflow",
        "objective": "Inspect repository state and run a verification computation "
                     "before reporting readiness.",
        "categories": ["version_control", "computation"],
    },
]


def _gemini_config(tool_recs: list, store_cfg: dict = None) -> dict:
    cfg = {
        "node_host": dt.NODE_HOST_FROM_VM, "node_port": dt.NODE_PORT,
        "username": dt.ADMIN_USER,
        **dt.llm_config(),
    }
    if store_cfg:
        # Path A: Davinci discovers tools by listing the store's MCP machines.
        cfg["store"] = store_cfg
    else:
        # Fallback: a pre-resolved (schema-less) catalog.
        cfg["tools"] = [{"name": t["name"], "tool_id": t["tool_id"], "category": t["category"],
                         "risk": t["risk"], "description": t["description"],
                         "program_id": t["program_id"], "entity_id": t["entity_id"],
                         "function": t.get("function", "invoke"),
                         "requires_network": t.get("requires_network", False)}
                        for t in tool_recs]
    return cfg


# --------------------------------------------------------------------------- #
# Store (space) setup — Path A discovery substrate
# --------------------------------------------------------------------------- #

STORES_WASM = os.environ.get(
    "STORES_WASM",
    str((REPO.parent / "decillionai-server" / "wasm" / "stores.wasm")))


def setup_store(c: CasparSignalingClient, tool_recs: list) -> dict:
    """Deploy the stores miniapp, create a store, and add each tool as a machine
    in it (carrying its MCP manifest). Returns the store discovery config for
    Davinci, or ``{}`` if the stores creature isn't available."""
    wasm = Path(STORES_WASM)
    if not wasm.exists():
        warn(f"stores.wasm not found at {wasm} — skipping store (Path A unavailable)")
        return {}
    stores = dt.deploy_wasm_creature(c, "stores", wasm, entity_id="main")
    target = {"creature_id": stores["machine_id"], "program_id": stores["program_id"],
              "entity": "main"}

    # Create a store (space) for the Davinci tool catalog.
    resp = c.signal_miniapp(creature_id=target["creature_id"], program_id=target["program_id"],
                            entity="main", action="create",
                            payload={"isPublic": True, "persHist": False,
                                     "origin": "davinci-tools", "metadata": {"name": "davinci-tools"}})
    store_id = ""
    if resp.get("ok"):
        res = resp.get("result", {})
        store_id = res.get("storeId") or (res.get("host") or {}).get("storeId") or ""
    if not store_id:
        warn(f"could not create store: {resp}")
        return {}
    ok(f"created store {store_id}")

    # Add each deployed tool program to the store with its full MCP manifest.
    added = 0
    for t in tool_recs:
        meta = dt.tool_full_metadata(t["tool_id"])
        meta = dict(meta)
        meta["program_id"] = t["program_id"]
        meta["entity_id"] = t["entity_id"]
        r = c.signal_miniapp(creature_id=target["creature_id"], program_id=target["program_id"],
                             entity="main", action="addMachine", store_id=store_id,
                             payload={"storeId": store_id, "programId": t["program_id"],
                                      "machineId": t["machine_id"], "metadata": meta})
        if r.get("ok"):
            added += 1
        else:
            warn(f"addMachine {t['tool_id']} failed: {r.get('error')}")
    ok(f"added {added}/{len(tool_recs)} tools to store {store_id}")
    return {**target, "store_id": store_id}


def run_scenario(c: CasparSignalingClient, davinci: dict, tool_recs: list,
                 scenario: dict, store_cfg: dict = None) -> dict:
    """Signal the Davinci creature for one scenario and assert the outcome."""
    name = scenario["name"]
    # Only require categories we actually deployed, so the plan is satisfiable.
    have = {t["category"] for t in tool_recs}
    required = [cat for cat in scenario["categories"] if cat in have] or list(have)[:2]
    task = {"objective": scenario["objective"], "required_categories": required}
    config = _gemini_config(tool_recs, store_cfg=store_cfg)

    info(f"scenario '{name}': required={required}")
    vm_id = c.run_entity(davinci["program_id"], "davinci",
                         params={"task.json": json.dumps(task),
                                 "config.json": json.dumps(config)},
                         ram_mb=512, max_exec_seconds=240)
    found, logs = c.wait_for_vm_log(vm_id, "DAVINCI_RESULT", timeout=220, poll=4)
    texts = [_log_text(l) for l in logs]

    boot = next((t for t in texts if "DAVINCI_BOOT" in t and "capabilities" in t), "")
    provider = "unknown"
    discovery = "unknown"
    if boot:
        try:
            caps = json.loads(boot.split("DAVINCI_BOOT", 1)[1]).get("capabilities", {})
            provider = caps.get("llm_provider", "unknown")
            discovery = caps.get("discovery", "unknown")
        except Exception:
            pass
    # Discovery sentinel: how many MCP machines Davinci listed from the store.
    disc_line = next((t for t in texts if "DAVINCI_DISCOVERY" in t), "")
    discovered = 0
    if disc_line:
        try:
            discovered = json.loads(disc_line.split("DAVINCI_DISCOVERY", 1)[1]).get("registered", 0)
        except Exception:
            pass
    gemini_calls = sum(1 for t in texts if t.lstrip().startswith("GEMINI ")
                       and '"propose"' in t)
    # Creature-to-creature evidence: the agent's executor records one
    # ``tool_result`` trace event per sibling tool-creature VM it signals (the
    # tool's own ``TOOL_RESPONSE`` line lives in the *tool* creature's logs and
    # is parsed into ``response``/``vm_id`` here, so we key on the trace event).
    c2c_calls = sum(1 for t in texts
                    if '"kind": "tool_result"' in t and '"vm_id"' in t)

    result = {}
    result_line = next((t for t in texts if "DAVINCI_RESULT" in t), "")
    if result_line:
        try:
            result = json.loads(result_line.split("DAVINCI_RESULT", 1)[1].strip())
        except Exception:
            result = {}

    tool_calls = int(result.get("budget", {}).get("tool_calls_used", 0) or 0)
    used_gemini = provider.startswith("gemini")
    # When a store is configured, discovery must have gone through Path A (the
    # store) and registered the MCP tools.
    disc_ok = (discovery == "store" and discovered > 0) if store_cfg else True
    # A scenario passes when Davinci produced a result, was driven by Gemini,
    # discovered its tools (via the store when configured), and signalled at
    # least one sibling tool creature creature-to-creature.
    passed = bool(found and result and used_gemini and disc_ok
                  and tool_calls > 0 and c2c_calls > 0)

    rec = {"scenario": name, "passed": passed, "vm_id": vm_id,
           "llm_provider": provider, "discovery": discovery, "discovered_tools": discovered,
           "gemini_propose_calls": gemini_calls,
           "tool_calls": tool_calls, "creature_to_creature_calls": c2c_calls,
           "success": result.get("success"),
           "steps_done": result.get("plan", {}).get("progress", {}).get("done")}
    (ok if passed else bad)(f"scenario '{name}': " + json.dumps(rec))
    if not passed:
        for t in texts[-8:]:
            print("    ", t[:160])
    return rec


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    if not dt.GEMINI_API_KEY:
        warn("GEMINI_API_KEY not set — the agent will fall back to the heuristic "
             "reasoner and Gemini assertions will fail. Pass the key to the "
             "workflow script.")

    tools = _select_tools()
    info(f"connecting to Caspar node {dt.NODE_HOST}:{dt.NODE_PORT}")
    c = CasparSignalingClient(dt.NODE_HOST, dt.NODE_PORT, timeout=90).connect()
    c.login(dt.ADMIN_USER)
    ok(f"logged in as {dt.ADMIN_USER} (user_id={c.user_id})")

    summary = {"tools_selected": [t["tool_id"] for t in tools],
               "tools_deployed": 0, "tools_signalled": 0,
               "davinci_deployed": False, "scenarios": []}

    info(f"── Phase 1: create creatures + programs and deploy {len(tools)} tool(s) ──")
    tool_recs = []
    for tool in tools:
        rec = dt.deploy_tool(c, tool)
        tool_recs.append(rec)
        summary["tools_deployed"] += 1

    info("── Phase 2: signal tool creatures directly ──")
    for rec in tool_recs:
        if dt.signal_tool(c, rec):
            summary["tools_signalled"] += 1

    info("── Phase 3: deploy stores miniapp, create store, add tools (Path A) ──")
    store_cfg = setup_store(c, tool_recs)
    summary["store_id"] = store_cfg.get("store_id", "")
    summary["discovery"] = "store" if store_cfg else "catalog"

    info("── Phase 4: create + deploy the Davinci agent creature ──")
    davinci = dt.deploy_davinci(c)
    summary["davinci_deployed"] = True

    info("── Phase 5: signal Davinci across diverse scenarios (Gemini-driven) ──")
    for scenario in SCENARIOS:
        # Skip scenarios whose categories we deployed none of.
        if not any(cat in {t["category"] for t in tool_recs} for cat in scenario["categories"]):
            continue
        summary["scenarios"].append(run_scenario(c, davinci, tool_recs, scenario, store_cfg))
        time.sleep(1)

    c.close()

    scen_pass = sum(1 for s in summary["scenarios"] if s["passed"])
    overall = (summary["tools_deployed"] == len(tools)
               and summary["tools_signalled"] == len(tools)
               and summary["davinci_deployed"]
               and summary["scenarios"]
               and scen_pass == len(summary["scenarios"]))
    summary["scenarios_passed"] = f"{scen_pass}/{len(summary['scenarios'])}"
    summary["overall"] = "PASS" if overall else "FAIL"

    print("\n" + "=" * 64)
    print("E2E_RESULT " + json.dumps(summary, indent=2))
    print("=" * 64)
    (ok if overall else bad)("OVERALL: " + summary["overall"])
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
