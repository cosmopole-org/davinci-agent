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
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import deploy_and_test as dt  # noqa: E402
from davinci.caspar_signaling import CasparSignalingClient, _log_text  # noqa: E402

GREEN, RED, CYAN, YELLOW, NC = dt.GREEN, dt.RED, dt.CYAN, "\033[0;33m", dt.NC


def info(m): print(f"{CYAN}[crypto-e2e]{NC} {m}", flush=True)
def ok(m):   print(f"{GREEN}[ ok ]{NC} {m}", flush=True)
def bad(m):  print(f"{RED}[fail]{NC} {m}", flush=True)
def warn(m): print(f"{YELLOW}[warn]{NC} {m}", flush=True)


TODAY = date.today()
_THIS_YEAR = TODAY.year

# The target trader's CoinMarketCap community profile is parameterisable via the
# CMC_PROFILE_URL env var, so any profile can be scored without editing this file.
# Defaults to the original vlad_anderson profile for backward compatibility.
_DEFAULT_CMC_PROFILE_URL = "https://coinmarketcap.com/community/profile/vlad_anderson/"
CMC_PROFILE_URL = (os.environ.get("CMC_PROFILE_URL") or "").strip() or _DEFAULT_CMC_PROFILE_URL
if not CMC_PROFILE_URL.endswith("/") and "?" not in CMC_PROFILE_URL and "#" not in CMC_PROFILE_URL:
    CMC_PROFILE_URL += "/"
# Derive the trader handle from the URL (.../community/profile/<handle>/).
_after_profile = CMC_PROFILE_URL.split("/profile/", 1)[-1]
CMC_TRADER_HANDLE = _after_profile.strip("/").split("/")[0].split("?")[0] or "the target trader"

TASK_OBJECTIVE = (
    f"Today's real-world date is {TODAY:%A, %d %B %Y} (i.e. the current year is "
    f"{_THIS_YEAR}). Use this as your reference 'now' for every date you resolve "
    """below.\n
    ROLE
You are an institutional-grade Financial Intelligence AI specialized in evaluating cryptocurrency and forex analysts.
Your task is NOT to predict the market.
Your task is to objectively measure the quality, accuracy, consistency and reliability of analysts based on their published posts and actual market performance.
Every score must be generated from verifiable market data.
Never guess.
If data is insufficient reduce confidence. you can use the browser tool to naviagte the web page. you can use the frtvh tool to fetch api driven data.

naviagte to the trader's coin market cap profile at https://coinmarketcap.com/community/profile/daniel_crypto365/ and do these based on that page:

INPUT
You receive
Analyst Information
username 
followers 
platform 
total posts 

Post
Each post contains
text 
timestamp 
symbol 
market 
images 
charts 
timeframe 
entry 
stoploss 
take profits 
prediction 
direction 
confidence (if exists) 

Market Data
Historical OHLCV
Real price
Volume
Funding Rate
Open Interest
Liquidation Data
CVD
Whale Transactions
Exchange Inflow/Outflow
On-chain Metrics
Fear & Greed
News
Economic Calendar
CPI
FOMC
Interest Rates
ETF Flows
DXY
Gold
NASDAQ
Correlation Data

OBJECTIVE
Generate a complete reliability profile for the analyst.
Do NOT evaluate writing quality.
Evaluate analytical performance.

STEP 1
Classify each post.
Categories
Technical Analysis
Fundamental Analysis
Mixed
News
Educational
Market Update
Signal
Trade Journal
Macro
Opinion
Ignore posts with no prediction.

STEP 2
Extract prediction.
Direction
Bullish
Bearish
Neutral
Range
Reversal
Breakout
Breakdown
Target
Stop
Invalidation
Time Horizon
Scalp
Intraday
Swing
Position
Long Term

STEP 3
Determine evaluation window.
If analyst explicitly says
Next 2 hours
Use exactly 2 hours.
If says
This week
Evaluate until week ends.
If says
Next month
Evaluate one month.
If no timeframe
Estimate from context.
Confidence decreases.

STEP 4
Measure actual market outcome.
Calculate
Maximum Favorable Excursion
Maximum Adverse Excursion
Final Return
Target Hit
Stop Hit
Drawdown
Volatility
ATR
Trend Strength
Relative Performance

STEP 5
Technical Analysis Evaluation
Evaluate
Trend Analysis
Support Resistance
Market Structure
Supply Demand
Liquidity
Order Blocks
FVG
Volume
Volume Profile
VWAP
EMA
SMA
MACD
RSI
Stochastic
ADX
Bollinger
Ichimoku
Fibonacci
Wyckoff
SMC
Price Action
Candlestick
Breakout Quality
Risk Reward
Entry Precision
Stop Placement
Target Placement
Probability
Score each
0-100

STEP 6
Fundamental Evaluation
Evaluate whether the fundamental reasoning was actually reflected in the market.
Examples
ETF approval
Interest rate
Fed speech
Inflation
Whale accumulation
Token unlock
Partnership
Hack
Exchange listing
Exchange delisting
SEC
Regulation
Protocol Upgrade
Burn
Tokenomics
On-chain activity
Developer Activity
TVL
Stablecoin Supply
Liquidity
Sentiment
News Impact
Macro
Dollar Index
Gold
NASDAQ
Oil
BTC Dominance
ETH Dominance
Funding Rate
Open Interest
Liquidations
Measure
Expected Impact
Actual Impact
Reaction Delay
Magnitude
Duration
Persistence
Noise
False Correlation
Assign score
0-100

STEP 7
Prediction Accuracy
For every prediction calculate
Correct Direction
Timing Accuracy
Entry Accuracy
Exit Accuracy
Maximum Profit
Maximum Loss
Risk Adjusted Return
Sharpe-like score
Precision
Confidence Calibration

STEP 8
Consistency
Measure
Winning Streak
Losing Streak
Monthly Accuracy
Quarterly Accuracy
Average Return
Median Return
Variance
Stability
Prediction Frequency
Confidence Stability

STEP 9
Reliability
Detect
Cherry Picking
Deleted Posts
Edited Posts
Contradictory Signals
Clickbait
Impossible Claims
Overconfidence
Prediction Spam
No Stop Loss
Always Bullish
Always Bearish
If detected
Decrease score.

STEP 10
Image Analysis
Analyze chart.
Identify
Trendlines
Indicators
Support
Resistance
Drawing Quality
Whether chart actually supports text.

STEP 11
Historical Performance
If analyst has many posts
Calculate
Last 10
Last 30
Last 50
Last 100
Lifetime
Accuracy
Average Return
Maximum Drawdown
Average Holding Time

STEP 12
Create Scores
Technical Accuracy
Technical Quality
Fundamental Accuracy
Fundamental Quality
Risk Management
Entry Quality
Exit Quality
Timing
Consistency
Transparency
Reliability
Market Understanding
Signal Strength
Overall Analyst Score
All scores
0-100

STEP 13
Analyst Rating
95-100
Elite Institutional
90-94
Professional
80-89
Advanced
70-79
Intermediate
60-69
Average
40-59
Weak
Below 40
Unreliable

STEP 14
Confidence
Every score must include
Confidence %
Based on
Sample Size
Market Data Completeness
Prediction Clarity
Timeframe Clarity
Historical Coverage

STEP 15
Final JSON
{
  "overall_score":0,
  "technical_score":0,
  "fundamental_score":0,
  "historical_accuracy":0,
  "consistency":0,
  "risk_management":0,
  "timing":0,
  "confidence":0,
  "technical_analysis":{
  },
  "fundamental_analysis":{
  },
  "historical_statistics":{
  },
  "market_validation":{
  },
  "prediction_validation":{
  },
  "strengths":[
  ],
  "weaknesses":[
  ],
  "warnings":[
  ]
}
"""
)

TOOL_IDS = ["web_search", "fetch_url", "browser_automation", "python_exec"]

# Long enough that every tool VM stays alive serving signals for the whole
# duration of the Davinci run; the tool also self-exits after this idle window.
TOOL_SERVE_SECONDS = 2000
DAVINCI_SERVE_SECONDS = 2000

# Per-tool serving-VM resources + how long to wait for TOOL_SERVE_READY.
#
# ROOT CAUSE of "tool browser_automation FAILED to serve" (the disk_gb field):
# the Caspar node creates each creature container with a per-container docker
# disk quota — ``HostConfig.StorageOpt = {"size": "<disk_gb>G"}`` — and docker
# requires that quota to be >= the image's virtual size, otherwise
# ``POST /containers/create`` is rejected. browser_automation ships the
# ~3.6 GB ``mcr.microsoft.com/playwright/python`` image, so the default
# disk_gb=1 (a 1 GB quota) makes its container impossible to create on any
# storage driver that enforces storage_opt, while the slim-python tools
# (images < 1 GB) fit and serve fine — exactly the "only the browser fails"
# symptom. The node creates containers fire-and-forget and swallows the 400, so
# it only ever surfaces as the opaque "FAILED to serve". (The node's prebuilt
# binary applies this quota even when CASPAR_DOCKER_DISK_QUOTA=0 is set, so the
# fix has to come from the resource request here: give the browser a quota
# comfortably larger than its image.)
#
# The browser VM also needs real RAM to launch headless Chromium once serving
# (the smoke-signal below and the real Davinci-driven run) — 320 MB OOMs it —
# and a longer ready window, since under gVisor the gofer serving that large
# rootfs can take a while to boot. The light tools keep the lean defaults so the
# run stays within the CI runner's memory budget.
_DEFAULT_SERVE = {"ram_mb": 320, "disk_gb": 1, "ready_timeout": 150}
TOOL_SERVE_RESOURCES = {
    # disk_gb=8 → an 8 GB storage_opt quota, well above the ~3.6 GB image.
    "browser_automation": {"ram_mb": 1536, "disk_gb": 8, "ready_timeout": 300},
}
# Davinci's internal wall-clock budget. Kept below the serve/await windows above
# so the agent finishes and replies before its serving VM is torn down. Raised
# from the default because the AgentRouter backbone adds latency per LLM call.
DAVINCI_MAX_WALL_SECONDS = 1800

STORES_WASM = os.environ.get(
    "STORES_WASM",
    str((REPO.parent / "decillionai-server" / "wasm" / "stores.wasm")))


# --------------------------------------------------------------------------- #
# Phase 2 — start tools as standalone serving VMs + smoke-signal each
# --------------------------------------------------------------------------- #

def _classify_serve_failure(texts: list) -> str:
    """Turn a serving VM's log tail into a one-line, actionable reason.

    The tool runtime emits explicit markers (TOOL_SERVE_BEGIN /
    TOOL_SERVE_UNAVAILABLE / TOOL_BRIDGE / TOOL_RESPONSE); read them so the
    harness reports *why* a tool never reached TOOL_SERVE_READY instead of the
    opaque "FAILED to serve".
    """
    nonblank = [t for t in texts if t.strip()]
    joined = "\n".join(texts)
    # The node emits these into the VM log stream when docker rejects the
    # container (e.g. a storage_opt disk quota smaller than the image, an unknown
    # runtime, or out-of-space) — the real, actionable root cause.
    for marker in ("CONTAINER_CREATE_FAILED", "CONTAINER_START_FAILED"):
        if marker in joined:
            line = next(t for t in texts if marker in t)
            return "node could not start the container → " + line.strip()[:240]
    if not nonblank:
        return ("container produced NO logs at all — the node never started it and "
                "did not report why. Most likely the docker storage_opt disk quota "
                "(HostConfig.StorageOpt.size = <disk_gb>G) is smaller than this "
                "tool's image; raise its disk_gb in TOOL_SERVE_RESOURCES above the "
                "image size. Check the node log / `docker events` for the real cause")
    # A concrete bridge connect/identity failure is the most specific root cause
    # (the node could not bind the container's source IP, the gateway address was
    # wrong, etc.) — surface it ahead of the generic "unavailable" summary.
    if "connect_error" in joined or "could not identify" in joined:
        line = next(t for t in texts if "connect_error" in t or "could not identify" in t)
        return "bridge connect/identity error → " + line.strip()[:240]
    if "TOOL_SERVE_UNAVAILABLE" in joined:
        line = next(t for t in texts if "TOOL_SERVE_UNAVAILABLE" in t)
        return "gateway bridge unavailable → " + line.split("TOOL_SERVE_UNAVAILABLE", 1)[-1].strip()
    if "TOOL_SERVE_BEGIN" not in joined:
        return ("runtime never reached main()/TOOL_SERVE_BEGIN — the container is "
                "not running tool_runtime.py (wrong CMD/entrypoint or import crash)")
    if "TOOL_RESPONSE" in joined and "TOOL_SERVE_READY" not in joined:
        return "tool ran one-shot offline (no bridge) and exited without entering the serve loop"
    if "Traceback" in joined or "Error" in joined:
        line = next(t for t in texts if "Traceback" in t or "Error" in t)
        return "tool crashed → " + line.strip()[:240]
    return ("timed out waiting for TOOL_SERVE_READY — TOOL_SERVE_BEGIN was logged "
            "but the bridge handshake never completed (still booting or stuck)")


def start_tool_serving(c: CasparSignalingClient, rec: dict) -> bool:
    res = TOOL_SERVE_RESOURCES.get(rec["tool_id"], _DEFAULT_SERVE)
    vm_id = c.run_entity(rec["program_id"], rec["entity_id"], params={},
                         ram_mb=res["ram_mb"], disk_gb=res["disk_gb"],
                         max_exec_seconds=TOOL_SERVE_SECONDS)
    rec["serve_vm"] = vm_id
    found, logs = c.wait_for_vm_log(vm_id, "TOOL_SERVE_READY",
                                    timeout=res["ready_timeout"], poll=3)
    if found:
        ok(f"tool {rec['tool_id']} serving (vm={vm_id})")
        return True
    # Surface the actual cause — a bare "FAILED to serve" is un-actionable.
    texts = [_log_text(l) for l in logs]
    bad(f"tool {rec['tool_id']} FAILED to serve (vm={vm_id}): {_classify_serve_failure(texts)}")
    tail = [t for t in texts if t.strip()][-15:]
    if tail:
        warn(f"{rec['tool_id']} serving-VM log tail ({len(tail)} lines):")
        for t in tail:
            print(f"      | {t[:240]}", flush=True)
    return False


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
        # Provider-neutral LLM backbone (Gemini/Anthropic/OpenAI/Grok via env).
        **dt.llm_config(),
        "tools": [
            {"name": t["name"], "tool_id": t["tool_id"], "category": t["category"],
             "risk": t["risk"], "description": t["description"],
             "program_id": t["program_id"], "entity_id": t["entity_id"],
             "machine_id": t["machine_id"], "function": t.get("function", "invoke"),
             "arg_schema": dt.tool_arg_schema(t["tool_id"]),
             "requires_network": t.get("requires_network", False)}
            for t in tool_recs],
    }
    config["max_wall_seconds"] = DAVINCI_MAX_WALL_SECONDS
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
    if not dt.llm_config().get("gemini_api_key") and not dt.llm_config().get("llm_provider"):
        bad("no LLM key set — set GEMINI_API_KEY (or LLM_PROVIDER + <PROVIDER>_API_KEY)")
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
