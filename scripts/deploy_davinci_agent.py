#!/usr/bin/env python3
"""deploy_davinci_agent.py — deploy ONLY the Davinci agent creature onto a
already-running Caspar node and print its program id.

This is the lean deploy path used by the Decillion e2e script to wire a **real**
Davinci docker agent as the backend's proxy target (as opposed to the offline
``davinciStub``). Unlike ``e2e_test.py`` it does not deploy the tool creatures
or run prompt scenarios — the Decillion backend supplies each agent's tool
catalog from its space at prompt time — so this stays fast and needs no live
LLM round-trip to succeed.

The LLM backbone credentials (``GEMINI_API_KEY`` / ``LLM_PROVIDER`` + its key,
read from the environment) are **baked into the creature image** so the agent
reasons with a real provider even though the backend proxy signals it without an
LLM config. Keys are read from the environment only; never written to disk here.

After deploying, it also starts the agent as a standalone running VM entity on
Caspar via the runEntity endpoint (so it is live and serving signals, not just
deployed). Disable with DAVINCI_RUN_ENTITY=0.

The started VM serves prompts persistently: caspar_runtime stays in a serve loop
(DAVINCI_SERVE_FOREVER, default on) handling every task signal until the node's
exec cap reaps it, and the node routes each backend prompt to the live VM
(push_signal_to_machine) instead of cold-spawning — so one warm agent handles
all prompts. Pair with DAVINCI_VM_MAX_SECONDS=unlimited for a never-reaped VM.

Connection (plaintext TCP, matching the casparctl local node):
    CASPAR_NODE_HOST   node host   (default 127.0.0.1)
    CASPAR_NODE_PORT   node TCP    (default 8074)
    CASPAR_CA_BUNDLE   CA bundle baked into the image for egress TLS
                       (default /etc/ssl/certs/ca-certificates.crt)

runEntity (standalone VM) knobs:
    DAVINCI_RUN_ENTITY     1 to start the VM after deploy (default), 0 to skip
    DAVINCI_VM_RAM_MB      VM RAM in MB        (default 512)
    DAVINCI_VM_DISK_GB     VM disk in GB       (default 1)
    DAVINCI_VM_CPUS        VM CPU cores        (default 1)
    DAVINCI_VM_MAX_SECONDS VM max exec seconds (default UNLIMITED => ~317y, since
                           the node has no true-unlimited cap; set a positive
                           integer to impose a finite cap)

Output (stdout, machine-readable — the caller greps these):
    DAVINCI_PROGRAM_ID=<id>
    DAVINCI_ENTITY_ID=davinci
    DAVINCI_VM_ID=<vmId>        (when the standalone runEntity start succeeds)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deploy_and_test as dt  # noqa: E402  (canonical deploy helpers)
from davinci.caspar_signaling import CasparSignalingClient  # noqa: E402


def _truthy(val: str) -> bool:
    return str(val).strip().lower() not in ("", "0", "false", "no", "off")


# The node has no true "unlimited" exec cap: /programs/runEntity clamps
# maxExecTimeSeconds <= 0 to 60, and the docker VM controller always spawns a
# reaper that stops the container after that many seconds. So "unlimited" is
# expressed as a very large (but i64/u64-safe) value — ~317 years — which the
# reaper effectively never reaches.
_VM_MAX_UNLIMITED = 10_000_000_000


def _vm_max_seconds() -> int:
    # Unlimited by default: the standalone davinci VM the deploy fires is meant to
    # be a persistent, always-warm agent, so it should never be time-reaped unless
    # the operator explicitly sets a finite DAVINCI_VM_MAX_SECONDS.
    raw = os.environ.get("DAVINCI_VM_MAX_SECONDS", "unlimited").strip().lower()
    if raw in ("0", "-1", "none", "inf", "infinite", "unlimited", "immortal", "forever"):
        return _VM_MAX_UNLIMITED
    try:
        val = int(raw)
    except ValueError:
        return _VM_MAX_UNLIMITED
    return _VM_MAX_UNLIMITED if val <= 0 else val


def main() -> int:
    host = os.environ.get("CASPAR_NODE_HOST", dt.NODE_HOST)
    port = int(os.environ.get("CASPAR_NODE_PORT", str(dt.NODE_PORT)))
    dt.info(f"connecting to Caspar node {host}:{port}")
    c = CasparSignalingClient(host, port, timeout=120).connect()
    c.login(dt.ADMIN_USER)
    dt.ok(f"logged in as {dt.ADMIN_USER} (user_id={c.user_id})")

    # Stop mode: gracefully stop a running davinci entity, then exit. Used by the
    # e2e to bring the davinci VM down cleanly BEFORE the Caspar node is stopped
    # (rather than yanked with the node). Best-effort — a not-running entity is
    # fine; the caller ignores failures.
    stop_pid = os.environ.get("DAVINCI_STOP_PROGRAM_ID", "").strip()
    if stop_pid:
        entity = os.environ.get("DAVINCI_ENTITY_ID", "davinci").strip() or "davinci"
        dt.info(f"stopping davinci entity {entity} on program {stop_pid} (graceful pre-shutdown)")
        try:
            c.stop_entity(stop_pid, entity)
            dt.ok(f"stopEntity requested for {stop_pid}/{entity}")
            print(f"DAVINCI_STOPPED={stop_pid}", flush=True)
        except Exception as exc:  # noqa: BLE001 — the entity may not be running
            dt.warn(f"stopEntity failed ({exc}); the entity may not be running")
        c.close()
        return 0

    bake = dt.llm_bake_env()
    if bake:
        provider = bake.get("LLM_PROVIDER", "gemini")
        dt.info(f"baking LLM backbone into the creature image (provider={provider})")
    else:
        dt.warn("no LLM key in the environment — the agent will fall back to the "
                "heuristic reasoner (set GEMINI_API_KEY for real reasoning)")

    # Reuse mode: redeploy a FRESH davinci entity onto an EXISTING program instead
    # of minting a new creature/program. Set DAVINCI_REUSE_PROGRAM_ID to the
    # program id already recorded in the manifest. A Caspar program is permanent
    # and re-deployable, so the updated entity (new Dockerfile + bundle) lands on
    # the SAME program id — picking up code changes while every already-deployed
    # agent proxy keeps targeting a valid davinci (no orphaning).
    reuse_pid = os.environ.get("DAVINCI_REUSE_PROGRAM_ID", "").strip()
    entity = os.environ.get("DAVINCI_ENTITY_ID", "davinci").strip() or "davinci"
    if reuse_pid:
        davinci = dt.deploy_davinci(c, bake_env=bake, program_id=reuse_pid, entity_id=entity)
    else:
        davinci = dt.deploy_davinci(c, bake_env=bake)

    # Machine-readable markers for the caller.
    print("DAVINCI_PROGRAM_ID=" + davinci["program_id"], flush=True)
    print("DAVINCI_ENTITY_ID=" + davinci["entity_id"], flush=True)

    # Also start it as a standalone running VM entity on Caspar (runEntity), so
    # the agent is live and serving signals rather than only deployed. Enabled by
    # default; set DAVINCI_RUN_ENTITY=0 to skip. Resources are tunable via env.
    if _truthy(os.environ.get("DAVINCI_RUN_ENTITY", "1")):
        ram = int(os.environ.get("DAVINCI_VM_RAM_MB", "512"))
        disk = int(os.environ.get("DAVINCI_VM_DISK_GB", "1"))
        cpus = int(os.environ.get("DAVINCI_VM_CPUS", "1"))
        max_s = _vm_max_seconds()
        max_label = "unlimited (~317y)" if max_s == _VM_MAX_UNLIMITED else f"{max_s}s"
        dt.info(f"starting davinci as a standalone VM entity via runEntity "
                f"(ram={ram}MB disk={disk}GB cpu={cpus} maxExec={max_label})")
        try:
            vm_id = c.run_entity(davinci["program_id"], davinci["entity_id"],
                                 ram_mb=ram, disk_gb=disk, cpu_cores=cpus,
                                 max_exec_seconds=max_s)
            if vm_id:
                dt.ok(f"davinci VM entity running: {vm_id}")
                print("DAVINCI_VM_ID=" + vm_id, flush=True)
            else:
                dt.warn("runEntity returned no vmId")
        except Exception as exc:  # non-fatal: the program is deployed regardless
            dt.warn(f"runEntity failed ({exc}); the program is deployed and the "
                    "backend will still spawn davinci per prompt")
    else:
        dt.info("DAVINCI_RUN_ENTITY=0 — skipping the standalone runEntity start")

    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
