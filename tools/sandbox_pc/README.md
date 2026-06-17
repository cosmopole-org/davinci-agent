# sandbox_pc

A davinci tool that gives the agent a private, **per-session sandbox VM** — one
Caspar VM (`fire` Firecracker microVM or `docker` gVisor container) bonded to the
session, whose installed software and data persist between turns.

## How it works

The tool runs as a Caspar `docker` creature. It manages the sandbox VM
**directly** through the node's vmm host functions over the docker-host bridge —
there is no intermediary creature to proxy through:

```
sandbox_pc tool (docker creature)
        │  bridge.call("runVm" | "execVm" | "terminateVm" | "copyToVm", …)
        ▼
Caspar node vmm  ──►  fire_vm_controller   (Firecracker microVM)
                 └─►  docker_vm_controller (gVisor container)
                          persistent disk: {STORAGE_ROOT_PATH}/vms/<vm>
```

All of the management logic lives in `davinci/sandbox_core.py`
(`SandboxManager`), which the davinci runtime's session-lifecycle hook
(`_SessionSandbox`) uses too — so the same session VM is addressed identically
whether it is being kept warm by the runtime or driven by the agent's tool call.

## Bonding & addressing

Each session keeps exactly one VM, keyed by `(machineId, vmId)`:

- `vmId = sess-<session_id>` bonds the VM to the session; a later prompt in the
  same session resumes it.
- `machineId` is a fixed namespace constant (`SANDBOX_PC_MACHINE`, default
  `sandbox`) — deliberately not the calling creature's identity — so the tool
  and the lifecycle hook (different creatures) resolve the same VM.

The node keys the persistent disk on `vmId` alone (`{storage}/vms/<vm>`), so the
disk survives suspend/resume. A VM is not bound 1:1 with a store/space — exactly
like the other VM types.

## Lifecycle

While the agent loop is actively servicing a session, the runtime `start`s /
keeps the VM warm. Between turns it `suspend`s the VM: the guest is stopped but
its disk survives, so the next `exec` resumes the same sandbox with all installed
software and saved data intact. The host implements **wake-on-exec**, so a
missed `start` is harmless — `exec` boots the VM automatically.

`delete` is the only destructive action: it stops the VM **and** purges its disk.

## Functions

| Function  | Effect                                                                 |
|-----------|------------------------------------------------------------------------|
| `exec`    | Run a shell command in the session VM; auto-wakes if suspended         |
| `start`   | Boot or resume the session VM on its persistent disk                   |
| `write`   | Drop a file into the session VM (`/data/<file_name>` by default)       |
| `suspend` | Turn the VM off but keep its disk (default idle state)                 |
| `delete`  | Turn the VM off and purge its disk (destroy the sandbox)               |
| `status`  | Report the VM's last-known state from host storage                     |

Each call accepts an optional `session_id` (defaults to the bridge handshake
session) and `runtime` (`fire` or `docker`).

## Configuration

- `SANDBOX_PC_RUNTIME` — default VM runtime (`fire` or `docker`). Pick `docker`
  on hosts without KVM. Set it on **both** the davinci runtime and the tool image
  so the lifecycle hook and the tool agree.
- `SANDBOX_PC_MACHINE` — namespace constant for the VM container/process key
  (default `sandbox`); must match wherever the session VM is addressed.
- `SANDBOX_PC_IMAGE` — base image for the docker runtime (default `alpine:latest`).
- The Caspar node must have `STORAGE_ROOT_PATH` configured; sandbox disks live
  under `${STORAGE_ROOT_PATH}/vms/`. For a real `fire` guest boot the node also
  needs `FIRECRACKER_KERNEL_IMAGE` and `FIRECRACKER_ROOTFS_IMAGE`.

## Safety

The sandbox directory is canonicalised and containment-checked by the node on
every `runVm`/`terminateVm`: paths that resolve outside `${STORAGE_ROOT_PATH}/vms`
are refused, so a crafted `vmId` cannot escape the sandbox tree. Inside a `fire`
guest the only block devices are the session's own rootfs + data disk — the host
filesystem is unreachable. `risk_level` is `high` (arbitrary shell), so davinci
routes the tool through its permission engine.
