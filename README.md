# Davinci Agent

Davinci Agent is an orchestration-first coding agent designed to plan work in a
central reasoning runtime and delegate execution to specialized tools.

Instead of acting as a single monolithic process, Davinci coordinates multiple
purpose-built tool containers (for git actions, web search, URL fetching,
browser automation, Python execution, connectors, and more). This design keeps
tasks modular, improves operational isolation, and makes it easier to evolve or
swap capabilities independently.

## What this agent does

Davinci Agent:

- receives user tasks and breaks them into actionable steps,
- selects the right tool/function for each step,
- sends structured requests through signal points,
- collects tool responses from expected response points,
- combines results into a coherent answer or action.

At a high level, Davinci acts as the **planner/brain**, while the tool
containers act as **specialized executors**.

## Runtime model

The runtime follows two key rules:

1. **Unified git tool access**  
   Git-related operations are routed through one tool container (`git_tool`) via
   multiple functions.

2. **Pre-running tools**  
   Davinci does not start tool containers itself. Tools are expected to be
   running and reachable before task processing begins.

For each tool invocation, Davinci emits one signal to the tool request point
and includes:

- the task payload,
- the target function,
- an `expectedResponsePoint` where the tool should reply.

## Tooling surface

Current tool-container folders include:

- `tools/git_tool`
- `tools/web_search`
- `tools/fetch_url`
- `tools/browser_automation`
- `tools/vector_search`
- `tools/sql_query`
- `tools/python_exec`
- `tools/slack_connector`
- `tools/jira_connector`
- `tools/calendar_connector`

## Run

```bash
python3 agentic_runtime.py
```

## Test

```bash
python3 -m pytest -q
```
