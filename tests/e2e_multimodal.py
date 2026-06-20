#!/usr/bin/env python3
"""End-to-end multimodal Davinci scenario.

Builds on scripts/deploy_and_test.py + scripts/e2e_test.py to drive the full
stack with a real multimodal prompt:

  1. deploy a small, diverse set of tool creatures (incl. sandbox_pc),
  2. deploy the Decillion ``stores`` miniapp + create a space, add tools to it,
  3. deploy the davinci agent creature,
  4. signal davinci with the user prompt PLUS an image attachment,
  5. capture every interaction (boot, attachments, gemini calls, tool calls,
     final answer) into a single report file.

Run inside the repo:  python3 tests/e2e_multimodal.py <image_path>
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

import deploy_and_test as dt  # noqa: E402
from davinci.attachments import attachment_from_file  # noqa: E402
from davinci.caspar_signaling import CasparSignalingClient, _log_text  # noqa: E402

CYAN, GREEN, RED, YELLOW, NC = dt.CYAN, dt.GREEN, dt.RED, "\033[0;33m", dt.NC

def info(m): print(f"{CYAN}[mm]{NC} {m}", flush=True)
def ok(m):   print(f"{GREEN}[ ok ]{NC} {m}", flush=True)
def bad(m):  print(f"{RED}[fail]{NC} {m}", flush=True)
def warn(m): print(f"{YELLOW}[warn]{NC} {m}", flush=True)


SELECTED_TOOLS = {"python_exec", "fetch_url", "vector_search", "sandbox_pc"}

STORES_WASM = Path(os.environ.get(
    "STORES_WASM",
    str((REPO.parent / "decillionai-server" / "wasm" / "stores.wasm"))))

REPORT_PATH = Path(os.environ.get(
    "MM_REPORT_PATH", str(REPO / "tests" / "_artifacts" / "e2e_multimodal_report.json")))


def select_tools() -> list:
    return [t for t in dt.TOOLS if t["tool_id"] in SELECTED_TOOLS]


def setup_store(c: CasparSignalingClient, tool_recs: list) -> dict:
    if not STORES_WASM.exists():
        warn(f"stores.wasm not at {STORES_WASM} — skipping Path A store discovery")
        return {}
    stores = dt.deploy_wasm_creature(c, "stores", STORES_WASM, entity_id="main")
    target = {"creature_id": stores["machine_id"], "program_id": stores["program_id"],
              "entity": "main"}
    resp = c.signal_miniapp(creature_id=target["creature_id"],
                            program_id=target["program_id"], entity="main",
                            action="create",
                            payload={"isPublic": True, "persHist": False,
                                     "origin": "davinci-mm",
                                     "metadata": {"name": "davinci-mm-tools"}})
    if not resp.get("ok"):
        warn(f"store creation failed: {resp}")
        return {}
    res = resp.get("result", {})
    store_id = res.get("storeId") or (res.get("host") or {}).get("storeId") or ""
    if not store_id:
        warn(f"no storeId in response: {res}")
        return {}
    ok(f"created store {store_id}")

    added = 0
    for t in tool_recs:
        meta = dict(dt.tool_full_metadata(t["tool_id"]))
        meta["program_id"] = t["program_id"]
        meta["entity_id"] = t["entity_id"]
        r = c.signal_miniapp(creature_id=target["creature_id"],
                             program_id=target["program_id"], entity="main",
                             action="addMachine", store_id=store_id,
                             payload={"storeId": store_id,
                                      "programId": t["program_id"],
                                      "machineId": t["machine_id"],
                                      "metadata": meta})
        if r.get("ok"):
            added += 1
            ok(f"added {t['tool_id']} to store")
        else:
            warn(f"addMachine {t['tool_id']} failed: {r.get('error')}")
    return {**target, "store_id": store_id, "added_count": added}


def signal_davinci_multimodal(c: CasparSignalingClient, davinci: dict,
                              tool_recs: list, store_cfg: dict,
                              image_path: str) -> dict:
    # The user's prompt + a real image attachment. attachment_from_file reads
    # the bytes, guesses the mime, and base64-encodes inline so the node ships
    # everything as one task.json upload to /app/input.
    attachment_spec = attachment_from_file(
        image_path,
        description="user-attached TradingView chart screenshot to analyze")
    task = {
        "objective": "analyze this picture and explain it",
        "session_id": "mm-demo-session",
        "attachments": [attachment_spec],
    }
    config = {
        "node_host": dt.NODE_HOST_FROM_VM, "node_port": dt.NODE_PORT,
        "username": dt.ADMIN_USER,
        **dt.llm_config(),
    }
    if store_cfg:
        config["store"] = store_cfg
    else:
        config["tools"] = [
            {"name": t["name"], "tool_id": t["tool_id"], "category": t["category"],
             "risk": t["risk"], "description": t["description"],
             "program_id": t["program_id"], "entity_id": t["entity_id"],
             "function": t.get("function", "invoke"),
             "requires_network": t.get("requires_network", False)}
            for t in tool_recs
        ]

    info(f"signalling davinci (image={Path(image_path).name},"
         f" {len(attachment_spec['data']) // 1000}kB base64)")
    vm_id = c.run_entity(davinci["program_id"], "davinci",
                         params={"task.json": json.dumps(task),
                                 "config.json": json.dumps(config)},
                         ram_mb=768, max_exec_seconds=300)
    info(f"davinci vm = {vm_id}; waiting for DAVINCI_RESULT…")
    found, logs = c.wait_for_vm_log(vm_id, "DAVINCI_RESULT",
                                     timeout=280, poll=4)
    texts = [_log_text(l) for l in logs]
    return {"vm_id": vm_id, "found_result": found, "logs": texts}


def harvest_tool_logs(c: CasparSignalingClient, tool_recs: list,
                      max_per_tool: int = 60) -> dict:
    """Pull the most recent TOOL_RESPONSE / log lines per tool creature."""
    out = {}
    for rec in tool_recs:
        try:
            logs = c.read_vm_logs(rec["program_id"], count=max_per_tool) or []
        except Exception as exc:  # noqa: BLE001
            out[rec["tool_id"]] = {"error": repr(exc)[:200]}
            continue
        texts = [_log_text(l) for l in logs]
        out[rec["tool_id"]] = {
            "tool_response_lines": [t for t in texts if "TOOL_RESPONSE" in t][-5:],
            "last_lines": texts[-10:],
        }
    return out


def summarize(davinci_run: dict) -> dict:
    texts = davinci_run["logs"]

    def _take_json(prefix: str):
        line = next((t for t in texts if prefix in t), "")
        if not line:
            return None
        try:
            return json.loads(line.split(prefix, 1)[1].strip())
        except Exception:
            return None

    boot = _take_json("DAVINCI_BOOT")
    attachments = _take_json("DAVINCI_ATTACHMENTS")
    bridge = _take_json("DAVINCI_BRIDGE")
    discovery = _take_json("DAVINCI_DISCOVERY")
    sandbox = [t for t in texts if "DAVINCI_SANDBOX" in t]
    gemini_lines = [t for t in texts if t.lstrip().startswith("GEMINI ")]
    propose_count = sum(1 for t in gemini_lines if '"propose"' in t and '"fallback"' not in t)
    reflect_count = sum(1 for t in gemini_lines if '"reflect"' in t and '"fallback"' not in t)
    upload_lines = [t for t in gemini_lines if "attachment_uploaded" in t
                    or "attachment_upload" in t]
    trace_lines = [t for t in texts if "DAVINCI_TRACE" in t]
    tool_result_traces = [t for t in trace_lines if '"kind": "tool_result"' in t]
    result = _take_json("DAVINCI_RESULT") or {}

    final_answer = result.get("answer") or ""
    return {
        "vm_id": davinci_run["vm_id"],
        "got_result": davinci_run["found_result"],
        "llm_provider": (boot or {}).get("capabilities", {}).get("llm_provider"),
        "discovery_mode": (boot or {}).get("capabilities", {}).get("discovery"),
        "attachments_seen_by_creature": attachments,
        "bridge_handshake": bridge,
        "discovery_event": discovery,
        "sandbox_lifecycle_events": sandbox,
        "gemini_propose_calls": propose_count,
        "gemini_reflect_calls": reflect_count,
        "gemini_attachment_upload_lines": upload_lines,
        "tool_result_trace_count": len(tool_result_traces),
        "tool_result_trace_preview": tool_result_traces[:6],
        "final_answer": final_answer,
        "success": result.get("success"),
        "steps": result.get("plan", {}).get("progress"),
        "budget": result.get("budget"),
    }


def main(image_path: str) -> int:
    if not dt.GEMINI_API_KEY:
        bad("GEMINI_API_KEY not set"); return 2
    if not Path(image_path).is_file():
        bad(f"image not found: {image_path}"); return 2

    tools = select_tools()
    info(f"selected tools: {[t['tool_id'] for t in tools]}")
    info(f"connecting to Caspar node {dt.NODE_HOST}:{dt.NODE_PORT}")
    c = CasparSignalingClient(dt.NODE_HOST, dt.NODE_PORT, timeout=600).connect()
    c.login(dt.ADMIN_USER)
    ok(f"logged in as {dt.ADMIN_USER} (user_id={c.user_id})")

    report: dict = {
        "image": {"path": image_path, "bytes": os.path.getsize(image_path)},
        "tools_selected": [t["tool_id"] for t in tools],
    }

    info("── Phase 1: deploy tool creatures ──")
    tool_recs = []
    for tool in tools:
        try:
            rec = dt.deploy_tool(c, tool)
            tool_recs.append(rec)
            ok(f"deployed {tool['tool_id']} (program={rec['program_id']})")
        except Exception as exc:  # noqa: BLE001
            bad(f"deploy {tool['tool_id']} failed: {exc}")
    report["tools_deployed"] = [{"tool_id": r["tool_id"],
                                 "program_id": r["program_id"],
                                 "machine_id": r["machine_id"]} for r in tool_recs]

    info("── Phase 2: signal tools directly to prove they answer ──")
    direct_results = {}
    for rec in tool_recs:
        try:
            direct_results[rec["tool_id"]] = dt.signal_tool(c, rec)
        except Exception as exc:  # noqa: BLE001
            direct_results[rec["tool_id"]] = False
            bad(f"direct signal {rec['tool_id']} failed: {exc}")
    report["tools_direct_smoke"] = direct_results

    info("── Phase 3: deploy stores miniapp, create space, add tools ──")
    store_cfg = setup_store(c, tool_recs)
    report["store"] = {"id": store_cfg.get("store_id", ""),
                       "added_count": store_cfg.get("added_count", 0)}

    info("── Phase 4: deploy davinci creature ──")
    davinci = dt.deploy_davinci(c)
    report["davinci"] = {"program_id": davinci["program_id"],
                         "machine_id": davinci["machine_id"]}

    info("── Phase 5: signal davinci with multimodal prompt ──")
    davinci_run = signal_davinci_multimodal(c, davinci, tool_recs,
                                            store_cfg, image_path)
    report["davinci_run"] = summarize(davinci_run)

    info("── Phase 6: harvest tool creature logs ──")
    report["tool_creature_logs"] = harvest_tool_logs(c, tool_recs)

    # Bound the full log: tail of last 200 lines is more than enough.
    report["davinci_full_logs_tail"] = davinci_run["logs"][-200:]

    c.close()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print("=" * 64)
    ok(f"report written: {REPORT_PATH}")
    final = report["davinci_run"].get("final_answer", "")[:600]
    info(f"final_answer (first 600 chars): {final or '(empty)'}")
    print("=" * 64)
    return 0 if davinci_run["found_result"] else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "tests/_artifacts/chart.jpg")
    sys.exit(main(path))
