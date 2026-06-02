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

```bash
# 1. bring up a single Caspar node (local binary mode, skip WASM creatures)
( cd ../caspar && ./run-nodes.sh single --no-docker --no-rebuild \
    --skip-deploy --no-gvisor --no-firecracker )

# 2. deploy tool creatures + the davinci creature, then test the full flow
python3 scripts/deploy_and_test.py
```

`deploy_and_test.py` performs four phases, all over the action protocol:

1. **Deploy tool creatures** — each tool (`web_search`, `vector_search`,
   `python_exec`, …) becomes its own `docker` creature; the node builds its image.
2. **Signal tools directly** — `runEntity` each tool, assert its `TOOL_RESPONSE`.
3. **Deploy the Davinci creature** — shipped as a tarball build context.
4. **Creature-to-creature** — signal Davinci with a task + tool catalog; Davinci
   logs in and signals the tool creatures *itself*, aggregating their responses.

### Tool-creature contract

A tool creature reads its signal from `/app/input/task.json`
(`{tool_id, function, payload}`) and emits one line `TOOL_RESPONSE <json>` on
stdout (captured as VM logs). See `tools/_runtime/tool_runtime.py`.

## Repository layout

```
davinci/                 # the agent runtime package
  engine.py              # the plan→reason→permit→act→reflect loop
  planning.py            # Plan / Planner / TODO tracking
  permissions.py         # risk tiers, modes, guardrails
  memory.py              # working / episodic / instruction memory
  observability.py       # tracer, budget, secret masking
  mcp.py                 # MCP-style tool registry
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
