# python_exec

Executes arbitrary Python code in an isolated subprocess with a wall-clock
timeout, CPU/address-space rlimits, captured stdout/stderr, and a `result`
variable convention.

- Image: `data-tools:latest`
- Function: `execute` (alias `invoke`)
- Build context: `tools/python_exec` (+ shared `tools/_runtime/tool_runtime.py`)

## Payload

| field          | type   | description                                  |
|----------------|--------|----------------------------------------------|
| `code`         | string | Python source to run (required)              |
| `stdin`        | string | fed to the process stdin                     |
| `timeout`      | int    | wall-clock seconds (default 30, max 300)     |
| `files`        | object | `{name: content}` written to the working dir |
| `requirements` | array  | pip packages installed before the run        |
| `env`          | object | extra environment variables                  |

Returns `{ok, returncode, result, vars, stdout, stderr, workdir_files}`.
