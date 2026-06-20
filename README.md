# Davinci Agent

Davinci is an **orchestration-first, enterprise-grade agent runtime**. It plans
work in a central reasoning loop, gates every action through a layered
permission/guardrail system, and delegates execution to specialized **tool
creatures** running as isolated VMs on a [Caspar](https://github.com/cosmopole-org/caspar)
node.

The brain (planning, permissions, budgeting, memory, tracing) is decoupled from
execution: the *same* engine drives an offline dry-run, a Caspar-signalling
deployment, or a real LLM backend by swapping two small plug-points.

## Architecture

```
            ┌──────────────────────────────────────────────┐
            │                 DavinciAgent                  │
            │  plan → reason → permit → act → reflect ↺      │
            ├──────────────┬───────────────┬────────────────┤
            │  Planner     │ PermissionEng │  Tracer/Budget │
            │ (plan+react) │ (risk+guards) │ (audit+limits) │
            ├──────────────┴───────────────┴────────────────┤
            │  Memory: working · episodic(JSONL) · DAVINCI.md │
            │  ToolRegistry (MCP-style, deferred schemas)     │
            └───────────────────────┬─────────────────────────┘
                                     │  ToolExecutor (pluggable)
                 ┌───────────────────┼────────────────────┐
                 ▼                   ▼                    ▼
          EchoExecutor      CasparSignalExecutor   CasparCreatureExecutor
          (offline)         (signal tool VMs)      (creature→creature)
```

## Enterprise features

Implemented in the `davinci/` package (stdlib-only, runs in a slim container):

| Area | Capability | Module |
|------|-----------|--------|
| Reasoning | plan-and-execute · ReAct · reflection · replan-on-failure · stuck/loop detection | `engine.py`, `planning.py` |
| LLM backbone | **provider-agnostic** reasoning/planning/loop · pluggable clients (Gemini · Anthropic/Claude · OpenAI · Grok · any OpenAI-compatible) · multi-model fallback | `reasoner.py`, `llm.py` |
| Permissions | risk tiers (low/med/high) · 6 permission modes · deny-first rules · input **and** output guardrails | `permissions.py` |
| Limits | bounded execution (steps/tool-calls) · token + cost budget · wall-clock deadline | `observability.py` |
| Memory | working memory · event-sourced JSONL episodic log (replay/resume) · hierarchical `DAVINCI.md` instructions · context compaction | `memory.py` |
| Tools | MCP-style registry · namespaced tools · **deferred schemas** + tool search | `mcp.py` |
| Observability | append-only structured trace · secret masking · trajectory summary | `observability.py` |
| Integration | Caspar signalling client (RSA-PSS) · creature-to-creature signalling | `caspar_signaling.py`, `caspar_runtime.py` |

These mirror the patterns used by Claude Code, OpenAI Agents SDK, LangGraph, and
OpenHands (see the feature survey that informed this design).

## Run locally

```bash
python3 -m davinci.cli "research and ship the feature" --mode auto
python3 -m davinci.cli --self-test          # capability snapshot
python3 -m pytest -q                         # 26 tests
```

## Deploy on Caspar (tool creatures + Davinci creature)

Davinci's tools are deployed as **separate Caspar `docker` creatures**, and the
Davinci creature interacts with them purely through the Caspar **signalling API**.
The agent's reasoning is driven by a **pluggable LLM backbone** — the same
reasoning/planning/loop algorithms run on **Gemini**, **Anthropic (Claude)**,
**OpenAI**, **Grok (xAI)**, or any OpenAI-compatible endpoint. Pick the provider
in the task config (`llm_provider` / `<provider>_api_key`, or an `llm` block);
Gemini is the default when only a Gemini key is supplied. The examples below pass
a Gemini key, but any provider's key works the same way.

### One-command end-to-end workflow (recommended)

`scripts/e2e_workflow.sh` is the single, reproducible entrypoint. It ensures
Docker + gVisor (`runsc`) + the `kasper` VM network are ready, brings up a
single Caspar node (**no built-in Decillion WASM creatures** — `--skip-deploy`),
deploys every tool **and** the agent as `docker`-container creatures, and signals
the agent across several diverse scenarios. The Gemini API key is passed as a
parameter and is **never written to disk or committed**.

```bash
# pass the Gemini API key as the first argument (or via $GEMINI_API_KEY)
./scripts/e2e_workflow.sh "<GEMINI_API_KEY>"

# options: --tools all | --tools python_exec,web_search | --keep-running |
#          --skip-node | --gemini-models a,b,c | --caspar-dir ../caspar
```

### Manual (two steps)

```bash
# 1. bring up a single Caspar node (local binary mode, skip WASM creatures).
#    gVisor stays ON: the node runs every creature under the runsc runtime.
( cd ../caspar && ./run-nodes.sh single --no-docker --no-rebuild \
    --skip-deploy --no-firecracker )

# 2. deploy tool creatures + the davinci creature, then test the full flow.
#    GEMINI_API_KEY selects the LLM backbone; the harness passes it to the
#    Davinci creature's config.json (never committed).
GEMINI_API_KEY="<key>" python3 scripts/e2e_test.py        # diverse scenarios
GEMINI_API_KEY="<key>" python3 scripts/deploy_and_test.py # single scenario
```

The agent reaches Gemini and the tools reach the network through the
environment's egress gateway: the harness bakes the host CA bundle into every
creature image so TLS verification succeeds inside the sandbox.

`deploy_and_test.py` performs four phases, all over the action protocol:

1. **Deploy tool creatures** — each tool (`web_search`, `vector_search`,
   `python_exec`, …) becomes its own `docker` creature; the node builds its image.
2. **Signal tools directly** — `runEntity` each tool, assert its `TOOL_RESPONSE`.
3. **Deploy the Davinci creature** — shipped as a tarball build context.
4. **Creature-to-creature** — signal Davinci with a task + tool catalog; Davinci
   logs in and signals the tool creatures *itself*, aggregating their responses.

### Multimodal prompts (file attachments)

Davinci accepts file attachments alongside the user prompt over the same
signalling API. Build a task with one or more `attachments` entries (each
`{name, mime_type, data | path, description}`) and ship it as `task.json`:

```python
from davinci import attachment_from_file
from davinci.caspar_signaling import CasparSignalingClient

with CasparSignalingClient(host, port) as c:
    c.login("user")
    c.run_entity_with_attachments(
        davinci_program_id, "davinci",
        objective="Describe what is in this image and crop the receipt.",
        attachments=["/path/to/photo.jpg", "/path/to/receipt.pdf"],
        config={"gemini_api_key": "..."},
    )
```

End-to-end:

1. The host harness reads each file and inlines its bytes as base64 in the
   task payload.
2. The Caspar node uploads `task.json` to `/app/input` inside the davinci
   docker creature.
3. The creature decodes each inline attachment and **writes it to its
   filesystem** at `/app/input/attachments/<name>` (where the agent and any
   delegated tool creature can read it as a real file).
4. The active LLM client sends the prompt with each attachment as a standard
   multimodal part. The `GeminiClient`, for example, uses a Gemini `parts[]`
   entry — inline (`inlineData`) for files
   ≤ 18 MiB, or uploaded via the **Gemini Files API** (`fileData`/`fileUri`)
   for larger files. This matches Google's official multimodal contract.

### Tool-creature contract

A tool creature reads its signal from `/app/input/task.json`
(`{tool_id, function, payload}`) and emits one line `TOOL_RESPONSE <json>` on
stdout (captured as VM logs). The shared runtime (`tools/_runtime/tool_runtime.py`)
loads the tool's own `tool.py` and dispatches to its `invoke()`.

### Docker-host bridge gateway (egress/ingress)

A docker creature is sandboxed with **no direct route to the outside world** —
its only channel is a single TCP connection to the Caspar node's *docker-host
bridge gateway*. The node injects only `CASPAR_GATEWAY_HOST`/`CASPAR_GATEWAY_PORT`
at container start — no identity and no secret. The node identifies the creature
from the connection's docker-network **source IP** (it asks docker which
container owns that IP, then maps the container name to the identity it recorded
at launch) and reports the resolved identity (`vmId`/`machineId`/`programId`/
`creatureId`) back in the handshake. A container can never declare or spoof its
own identity.

`davinci/caspar_bridge.py` (`CasparBridgeClient`, `bridge_from_env()`) speaks the
gateway's chunked wire protocol. Over that one connection a creature:

- runs any node host function — DB/storage ops, **outbound HTTP**, signalling —
  via `bridge.call(op, input)` (the node stamps the verified identity), and
- receives pushed signals from other creatures via `bridge.on_signal(handler)`.

When the gateway env is present the Davinci creature drives sibling tool
creatures over the bridge (`BridgeCreatureExecutor`): it signals a tool's
machine and awaits the tool's correlated reply — both pushed over their
respective gateway connections. The client is shipped into every tool image by
the deploy harness; the tool runtime auto-connects and replies over it. See
`caspar/docs/DOCKER_HOST_GATEWAY.md` for the protocol spec.

### Tools (real implementations)

Every tool is a full implementation, not a stub. Each ships its own
`Dockerfile` + `requirements.txt`; the deploy harness builds each image with the
tool's real dependencies and the shared runtime.

| tool | function | what it does | key deps / credentials |
|------|----------|--------------|------------------------|
| `python_exec` | `execute` | runs Python in an isolated subprocess (timeout, rlimits, captured stdio, `result`) | numpy, pandas |
| `web_search` | `search` | web search via Tavily/Brave/SerpApi/Google CSE, DuckDuckGo fallback | `TAVILY_API_KEY` etc. (optional) |
| `fetch_url` | `fetch` | HTTP fetch + readable HTML/JSON extraction | requests, beautifulsoup4 |
| `vector_search` | `vector_search` | persistent semantic index + cosine search | scikit-learn (hashing) / sentence-transformers / OpenAI |
| `sql_query` | `query` | SQL via SQLAlchemy (SQLite/Postgres/MySQL) | `DATABASE_URL` (defaults to local SQLite) |
| `git_tool` | `status_commit`, `open_pr` | real git status/commit + GitHub PR creation | git, `GITHUB_TOKEN` |
| `browser_automation` | `automate` | headless Chromium/Firefox/WebKit via Playwright | Playwright base image |
| `slack_connector` | `slack_action` | Slack Web API (post/history/upload/…) | `SLACK_BOT_TOKEN` |
| `jira_connector` | `jira_action` | Jira Cloud REST v3 (issues/JQL/transitions) | `JIRA_URL`/`JIRA_EMAIL`/`JIRA_API_TOKEN` |
| `calendar_connector` | `calendar_action` | Google Calendar / CalDAV / `.ics` generation | Google SA / CalDAV creds (ICS works offline) |

Connectors that need credentials return a clear `{ok: false, error}` when they
are absent — the call path itself is fully real.

## Repository layout

```
davinci/                 # the agent runtime package
  engine.py              # the plan→reason→permit→act→reflect loop
  planning.py            # Plan / Planner / TODO tracking
  permissions.py         # risk tiers, modes, guardrails
  memory.py              # working / episodic / instruction memory
  observability.py       # tracer, budget, secret masking
  mcp.py                 # MCP-style tool registry
  reasoner.py            # provider-agnostic LLMReasoner (shared reasoning logic)
  llm.py                 # cross-LLM clients: Gemini · Anthropic · OpenAI · Grok
  caspar_signaling.py    # Caspar TCP client (RSA-PSS signing)
  caspar_runtime.py      # docker-creature entrypoint (live signalling)
  caspar_executor.py     # signal pre-running Caspar tool VMs
  cli.py                 # local CLI
tools/                   # tool creatures (each a docker image)
  _runtime/              # shared tool-creature runtime + Dockerfile
scripts/deploy_and_test.py   # full deploy + signalling test harness
caspar_orchestrator.py   # legacy/low-level Caspar VMM orchestration model
agentic_runtime.py       # capability report model
```
