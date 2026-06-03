#!/usr/bin/env bash
#
# e2e_workflow.sh — one-command, reproducible end-to-end workflow for Davinci on
# Caspar. It performs the *entire* pipeline:
#
#   1. Ensure Docker + gVisor (runsc) + the ``kasper`` VM network are ready.
#   2. Bring up a single Caspar node via run-nodes.sh (local binary, no built-in
#      Decillion WASM creatures — ``--skip-deploy``).
#   3. Create a machine creature + ``docker`` program for the Davinci agent and
#      for each Davinci tool, and deploy them as docker-container VMs.
#   4. Use the Gemini API key passed to THIS script as the agent's LLM backbone.
#   5. Signal the tools directly and signal the Davinci agent across several
#      diverse scenarios (creature-to-creature), asserting the whole flow.
#
# The Gemini API key is taken as a parameter / env var only and is NEVER written
# to disk or committed.
#
# Usage:
#   ./scripts/e2e_workflow.sh <GEMINI_API_KEY> [options]
#   GEMINI_API_KEY=... ./scripts/e2e_workflow.sh [options]
#
# Options:
#   --gemini-api-key KEY   Gemini API key (alternative to the positional arg).
#   --gemini-models LIST   Comma-separated model fallback list
#                          (default: gemini-flash-latest,gemini-2.5-flash,gemini-2.5-pro).
#   --caspar-dir DIR       Path to the caspar repo (default: ../caspar).
#   --tools LIST|all       Tool creatures to deploy (default: a diverse subset).
#   --node-host-from-vm IP Address creatures use to reach the node
#                          (default: 172.18.0.1 — the kasper bridge gateway).
#   --keep-running         Leave the Caspar node running after the tests.
#   --skip-node            Assume a node is already running; skip bring-up.
#   -h, --help             Show this help.
set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAVINCI_DIR="$(dirname "$SCRIPT_DIR")"
CASPAR_DIR="$(cd "$DAVINCI_DIR/.." && pwd)/caspar"

# ─── Defaults ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY="${GEMINI_API_KEY:-}"
GEMINI_MODELS="${GEMINI_MODELS:-gemini-flash-latest,gemini-2.5-flash,gemini-2.5-pro}"
E2E_TOOLS="${E2E_TOOLS:-}"
NODE_HOST_FROM_VM="${CASPAR_NODE_HOST_FROM_VM:-172.18.0.1}"
CA_BUNDLE="${CASPAR_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
KEEP_RUNNING=false
SKIP_NODE=false

C_INFO="\033[0;36m"; C_OK="\033[0;32m"; C_WARN="\033[0;33m"; C_ERR="\033[0;31m"; C_NC="\033[0m"
info() { echo -e "${C_INFO}[workflow]${C_NC} $*"; }
ok()   { echo -e "${C_OK}[ ok ]${C_NC} $*"; }
warn() { echo -e "${C_WARN}[warn]${C_NC} $*"; }
err()  { echo -e "${C_ERR}[fail]${C_NC} $*" >&2; }
die()  { err "$*"; exit 1; }

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

# ─── Arg parsing ──────────────────────────────────────────────────────────────
POSITIONAL_KEY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gemini-api-key)     GEMINI_API_KEY="$2"; shift 2 ;;
    --gemini-models)      GEMINI_MODELS="$2"; shift 2 ;;
    --caspar-dir)         CASPAR_DIR="$2"; shift 2 ;;
    --tools)              E2E_TOOLS="$2"; shift 2 ;;
    --node-host-from-vm)  NODE_HOST_FROM_VM="$2"; shift 2 ;;
    --keep-running)       KEEP_RUNNING=true; shift ;;
    --skip-node)          SKIP_NODE=true; shift ;;
    -h|--help)            usage ;;
    -*)                   die "unknown option: $1" ;;
    *)                    POSITIONAL_KEY="$1"; shift ;;
  esac
done
[[ -z "$GEMINI_API_KEY" && -n "$POSITIONAL_KEY" ]] && GEMINI_API_KEY="$POSITIONAL_KEY"

[[ -n "$GEMINI_API_KEY" ]] || die "no Gemini API key — pass it as the first argument or via GEMINI_API_KEY"
[[ -d "$CASPAR_DIR" ]]     || die "caspar repo not found at $CASPAR_DIR (use --caspar-dir)"

PY="$(command -v python3 || true)"; [[ -n "$PY" ]] || die "python3 not found"

# ─── 0. Host harness dependency (signing client) ──────────────────────────────
if ! "$PY" -c "import Crypto" >/dev/null 2>&1; then
  info "installing pycryptodome (host signalling client dependency)…"
  "$PY" -m pip install --quiet pycryptodome >/dev/null 2>&1 \
    || die "failed to install pycryptodome"
fi

# ─── 1. Docker + gVisor (runsc) + kasper network ──────────────────────────────
# The node spawns each creature as a docker container with the ``runsc`` runtime
# on the ``kasper`` network. We make that available here so the workflow is
# self-contained even on hosts where run-nodes.sh cannot restart the daemon
# (no systemd/snap, e.g. the web runner). run-nodes.sh applies the same fix when
# it manages gVisor itself.
ensure_docker_runsc() {
  command -v docker >/dev/null 2>&1 || die "docker not installed"

  if ! command -v runsc >/dev/null 2>&1; then
    info "installing runsc (gVisor)…"
    local arch; arch="$(uname -m)"
    curl -fsSL "https://storage.googleapis.com/gvisor/releases/release/latest/${arch}/runsc" \
      -o /usr/local/bin/runsc
    chmod +x /usr/local/bin/runsc
  fi

  # Register runsc with the runtimeArgs required for nested / cgroup-less hosts.
  "$PY" - <<'PYEOF'
import json, os
p = "/etc/docker/daemon.json"
try:
    cfg = json.loads(open(p).read().strip() or "{}")
except FileNotFoundError:
    cfg = {}
want = {"path": "runsc",
        "runtimeArgs": ["--network=host", "--platform=ptrace", "--ignore-cgroups"]}
if cfg.get("runtimes", {}).get("runsc") != want:
    cfg.setdefault("runtimes", {})["runsc"] = want
    os.makedirs("/etc/docker", exist_ok=True)
    json.dump(cfg, open(p, "w"), indent=2)
    print("updated")
else:
    print("unchanged")
PYEOF

  # Apply daemon.json: prefer systemd; otherwise (re)start a plain dockerd.
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet docker 2>/dev/null; then
    if ! docker info 2>/dev/null | grep -q runsc; then
      info "restarting docker (systemd) to load runsc runtime…"
      systemctl restart docker
    fi
  else
    if ! docker info 2>/dev/null | grep -q runsc; then
      info "(re)starting dockerd to load runsc runtime…"
      pkill -f dockerd 2>/dev/null || true
      sleep 2
      setsid dockerd --host=unix:///var/run/docker.sock >/tmp/dockerd.e2e.log 2>&1 < /dev/null &
      disown || true
    fi
  fi

  local i
  for i in $(seq 1 40); do docker info >/dev/null 2>&1 && break; sleep 1; done
  docker info >/dev/null 2>&1 || die "docker daemon not reachable (see /tmp/dockerd.e2e.log)"
  for i in $(seq 1 10); do docker info 2>/dev/null | grep -q runsc && break; sleep 1; done
  if docker info 2>/dev/null | grep -q runsc; then
    ok "runsc runtime registered with Docker"
  else
    warn "runsc not visible in docker info — creatures may fail"
  fi

  if docker network inspect kasper >/dev/null 2>&1; then
    ok "docker network 'kasper' present"
  else
    docker network create kasper >/dev/null 2>&1 && ok "created docker network 'kasper'" \
      || warn "could not create docker network 'kasper'"
  fi
}

# ─── 2. Bring up the Caspar node ──────────────────────────────────────────────
start_node() {
  info "bringing up a single Caspar node (local binary, --skip-deploy)…"
  ( cd "$CASPAR_DIR" && ./run-nodes.sh single --no-docker --no-rebuild \
       --skip-deploy --no-firecracker ) || die "run-nodes.sh failed"

  info "waiting for node TCP API on 127.0.0.1:8074…"
  local i
  for i in $(seq 1 60); do
    "$PY" - <<'PYEOF' && break
import socket, sys
s = socket.socket(); s.settimeout(2)
try:
    s.connect(("127.0.0.1", 8074)); sys.exit(0)
except Exception:
    sys.exit(1)
PYEOF
    sleep 1
  done
  "$PY" - <<'PYEOF' || die "node did not open TCP 8074"
import socket, sys
s = socket.socket(); s.settimeout(2)
sys.exit(0 if not s.connect_ex(("127.0.0.1", 8074)) else 1)
PYEOF
  ok "Caspar node is up on 127.0.0.1:8074"
}

stop_node() {
  $KEEP_RUNNING && { info "leaving node running (--keep-running)"; return; }
  info "stopping Caspar node…"
  ( cd "$CASPAR_DIR" && ./stop-nodes.sh >/dev/null 2>&1 ) || true
}

# ─── Run ──────────────────────────────────────────────────────────────────────
ensure_docker_runsc
if ! $SKIP_NODE; then
  start_node
  trap stop_node EXIT
else
  info "--skip-node: assuming a Caspar node is already running on 8074"
fi

# ─── Build the Decillion stores miniapp (Path A discovery substrate) ──────────
# Davinci discovers tools by listing a store's MCP machines, so we build+deploy
# the stores creature. Only the stores creature is built (per request).
SERVER_DIR="${DECILLIONAI_SERVER_DIR:-$(cd "$CASPAR_DIR/.." && pwd)/decillionai-server}"
STORES_WASM="${STORES_WASM:-$SERVER_DIR/wasm/stores.wasm}"
ensure_stores_wasm() {
  [[ -f "$STORES_WASM" ]] && { ok "stores.wasm present ($STORES_WASM)"; return; }
  [[ -d "$SERVER_DIR/creatures/stores" ]] || {
    warn "decillionai-server stores creature not found at $SERVER_DIR — Path A discovery will be skipped"; return; }
  if ! command -v tinygo >/dev/null 2>&1 && [[ ! -x /usr/local/tinygo/bin/tinygo ]]; then
    warn "tinygo not installed — cannot build stores.wasm; Path A discovery will be skipped"; return
  fi
  info "building stores.wasm (TinyGo)…"
  ( export PATH="/usr/local/go123/bin:/usr/local/tinygo/bin:$PATH"; export GOROOT="/usr/local/go123"
    mkdir -p "$SERVER_DIR/wasm"
    cd "$SERVER_DIR/creatures/stores" && \
    tinygo build -o "$STORES_WASM" -target=wasi -no-debug -opt=2 -scheduler=none . ) \
    && ok "built $STORES_WASM" || warn "stores.wasm build failed — Path A discovery will be skipped"
}
ensure_stores_wasm

info "── running end-to-end test orchestrator ──"
export GEMINI_API_KEY GEMINI_MODELS E2E_TOOLS STORES_WASM
export CASPAR_NODE_HOST_FROM_VM="$NODE_HOST_FROM_VM"
export CASPAR_CA_BUNDLE="$CA_BUNDLE"

set +e
"$PY" "$SCRIPT_DIR/e2e_test.py"
rc=$?
set -e

if [[ $rc -eq 0 ]]; then
  ok "END-TO-END WORKFLOW PASSED"
else
  err "END-TO-END WORKFLOW FAILED (exit $rc)"
fi
exit $rc
