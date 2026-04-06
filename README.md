# Davinci Agent: Brain + Caspar VMM Host-Function Orchestration

Davinci is the planner/reasoner (**brain VM**) and delegates execution to
specialized worker tool containers through Caspar VMM host functions.

## Updated runtime model

Two important constraints are now enforced:

1. **Unified git tool**: git operations are routed through one tool container
   (`git_tool`) with multiple functions (`status_commit`, `open_pr`).
2. **No Davinci-managed tool startup**: tool containers are assumed to be
   already running before Davinci starts processing user tasks.

Davinci now only sends point signals to tools and expects responses back on
registered response points.

## Signaling flow

For each planned category, Davinci emits one signaling action:

1. `signalPoint` to tool request point (`tool::<id>::request`)
2. payload includes task, requested function, and `expectedResponsePoint`
3. tool VM responds by signaling the response point

There is no `runVm` deployment phase in the planner execution path.

## Tool container layout

Each tool remains isolated in its own folder with its own Dockerfile:

- `tools/git_tool` (unified git tool)
- `tools/web_search`
- `tools/fetch_url`
- `tools/browser_automation`
- `tools/vector_search`
- `tools/sql_query`
- `tools/python_exec`
- `tools/slack_connector`
- `tools/jira_connector`
- `tools/calendar_connector`

## Usage

```bash
python3 agentic_runtime.py
```

## Tests

```bash
python3 -m pytest -q
```
