#!/usr/bin/env python3
"""deploy_sandbox_tool.py — deploy ONLY the per-space sandbox tool creature onto
an already-running Caspar node and print the ids Nest needs to signal it.

This is the sibling of ``deploy_davinci_agent.py``: the Decillion CI deploys the
davinci agent, then this, and hands both sets of ids to the Nest deployer, which
records them in ``.caspar-deploy.json``. Nest then signals this creature to
create a sandbox whenever a space is created, to destroy it when the space is
deleted, and publishes it into the space so every agent there discovers it as a
tool.

The Vercel credentials (``VERCEL_TOKEN``, ``VERCEL_TEAM_ID``,
``VERCEL_PROJECT_ID``, …) are read from this process's environment and **baked
into the creature image**, so they never travel in a signal payload that an
agent's prompt could influence. They are never written to the repo.

Connection (plaintext TCP, matching the casparctl local node):
    CASPAR_NODE_HOST   node host   (default 127.0.0.1)
    CASPAR_NODE_PORT   node TCP    (default 8074)
    CASPAR_CA_BUNDLE   CA bundle baked in for egress TLS

Reuse / lifecycle knobs:
    SANDBOX_REUSE_PROGRAM_ID   redeploy onto this existing program id instead of
                               minting a new creature (what CI passes on every
                               run after the first, so the ids Nest stores stay
                               valid)
    SANDBOX_TOOL_ENTITY_ID     entity id (default vercel_sandbox)
    SANDBOX_RUN_ENTITY         1 (default) to start it serving after deploy
    SANDBOX_VM_RAM_MB          VM RAM in MB        (default 256)
    SANDBOX_VM_DISK_GB         VM disk in GB       (default 1)
    SANDBOX_VM_CPUS            VM CPU cores        (default 1)
    SANDBOX_VM_MAX_SECONDS     VM max exec seconds (default unlimited)

Output (stdout, machine-readable — the caller greps these):
    SANDBOX_TOOL_PROGRAM_ID=<id>
    SANDBOX_TOOL_CREATURE_ID=<id>        (empty on a redeploy onto an existing program)
    SANDBOX_TOOL_ENTITY_ID=vercel_sandbox
    SANDBOX_TOOL_VM_ID=<vmId>            (when the standalone runEntity start succeeds)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deploy_and_test as dt  # noqa: E402  (canonical deploy helpers)
from davinci.caspar_signaling import CasparSignalingClient  # noqa: E402

TOOL_ID = "vercel_sandbox"


def _truthy(val: str) -> bool:
    return str(val).strip().lower() not in ("", "0", "false", "no", "off")


def _tool_spec() -> dict:
    """The tool's entry in the auto-discovered catalog (from point.metadata.json)."""
    for tool in dt.TOOLS:
        if tool["tool_id"] == TOOL_ID:
            return tool
    raise SystemExit(f"[fail] no tools/{TOOL_ID}/point.metadata.json found")


def main() -> int:
    host = os.environ.get("CASPAR_NODE_HOST", dt.NODE_HOST)
    port = int(os.environ.get("CASPAR_NODE_PORT", str(dt.NODE_PORT)))
    dt.info(f"connecting to Caspar node {host}:{port}")
    c = CasparSignalingClient(host, port, timeout=120).connect()
    c.login(dt.ADMIN_USER)
    dt.ok(f"logged in as {dt.ADMIN_USER} (user_id={c.user_id})")

    bake = dt.sandbox_bake_env()
    if bake.get("VERCEL_TOKEN"):
        scope = bake.get("VERCEL_TEAM_ID") or "personal account"
        dt.info(f"baking Vercel credentials into the sandbox creature image (scope={scope})")
    else:
        dt.warn("no VERCEL_TOKEN in the environment — the sandbox creature will deploy "
                "but every call will fail until the token is baked in")

    entity = os.environ.get("SANDBOX_TOOL_ENTITY_ID", TOOL_ID).strip() or TOOL_ID
    reuse_pid = os.environ.get("SANDBOX_REUSE_PROGRAM_ID", "").strip()
    spec = dict(_tool_spec())
    if entity != TOOL_ID:
        dt.warn(f"entity id {entity} differs from the tool id; Nest must be told the same value")

    rec = dt.deploy_tool(c, spec, program_id=reuse_pid or None, bake_env=bake)

    print("SANDBOX_TOOL_PROGRAM_ID=" + rec["program_id"], flush=True)
    print("SANDBOX_TOOL_CREATURE_ID=" + (rec.get("machine_id") or ""), flush=True)
    print("SANDBOX_TOOL_ENTITY_ID=" + rec["entity_id"], flush=True)

    # Start it as a long-lived serving creature: the tool runtime stays in its
    # serve loop and answers every signal over the gateway, so Nest's create /
    # delete calls and the agents' exec calls all hit a warm container instead of
    # cold-spawning one per invocation.
    if _truthy(os.environ.get("SANDBOX_RUN_ENTITY", "1")):
        ram = int(os.environ.get("SANDBOX_VM_RAM_MB", "256"))
        disk = int(os.environ.get("SANDBOX_VM_DISK_GB", "1"))
        cpus = int(os.environ.get("SANDBOX_VM_CPUS", "1"))
        raw_max = os.environ.get("SANDBOX_VM_MAX_SECONDS", "unlimited").strip().lower()
        try:
            max_s = int(raw_max)
            if max_s <= 0:
                raise ValueError
        except ValueError:
            max_s = 10_000_000_000  # the node has no true "unlimited" cap; ~317y
        dt.info(f"starting {TOOL_ID} as a standalone serving VM (ram={ram}MB cpu={cpus})")
        try:
            vm_id = c.run_entity(rec["program_id"], rec["entity_id"], ram_mb=ram, disk_gb=disk,
                                 cpu_cores=cpus, max_exec_seconds=max_s, force_restart=True)
            if vm_id:
                dt.ok(f"{TOOL_ID} VM entity running: {vm_id}")
                print("SANDBOX_TOOL_VM_ID=" + vm_id, flush=True)
            else:
                dt.warn("runEntity returned no vmId")
        except Exception as exc:  # non-fatal: the program is deployed regardless
            dt.warn(f"runEntity failed ({exc}); the node will cold-spawn the tool per signal")
    else:
        dt.info("SANDBOX_RUN_ENTITY=0 — skipping the standalone runEntity start")

    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
