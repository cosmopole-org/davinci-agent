# sandbox_pc

A davinci tool that exposes a per-session Firecracker micro-VM ("fire VM") as a
persistent sandbox. The tool itself is a thin proxy: the actual VM management
runs in the `sandbox` WASM creature (see
`decillionai-server/creatures/sandbox`) on Caspar, which drives the `fire`
runtime through the Caspar host APIs (`runVm` / `execVm` / `terminateVm`).

## How it works

```
davinci  ──signal──►  sandbox WASM creature  ──hostCall──►  Caspar fire VM
   ▲                                                         (persistent
   └──────── result (correlationId match) ─────────          disk under
                                                              {storage}/vms/
                                                              <machine>/<vm>)
```

Each davinci session keeps its own fire VM, keyed by `session_id`. The VM is
owned by the sandbox creature that ran it — fire VMs are not bound 1:1 with a
store; a single creature (whether a member of a store or not) can run unlimited
fire VMs in parallel, each in its own non-escapable directory under
`{STORAGE_ROOT_PATH}/vms/<machine>/<vm>`.

## Lifecycle

While the agent loop is actively servicing a session, it calls `start` to wake
or keep the VM warm. Between turns (waiting for the user's next prompt) it
calls `suspend`: the guest process is stopped but its rootfs and data disk
survive, so the next `exec` resumes the same sandbox with all installed
software and saved data intact. The host implements wake-on-exec, so a missed
`start` is harmless — `exec` will boot the VM automatically.

`delete` is the only destructive action: it stops the VM **and** purges its
persistent directory.

## Functions

| Function  | Effect                                                                 |
|-----------|------------------------------------------------------------------------|
| `exec`    | Run a shell command in the session VM; auto-wakes if suspended         |
| `start`   | Boot or resume the session VM on its persistent disk                   |
| `write`   | Drop a file into the session VM (`/tmp/<file_name>` by default)        |
| `suspend` | Turn the VM off but keep its disk (default idle state)                 |
| `delete`  | Turn the VM off and purge its disk (destroy the sandbox)               |
| `status`  | Report the VM's last-known state from host storage                     |

## Configuration

- `SANDBOX_PC_TARGET` — machine id of the deployed sandbox creature (default
  `sandbox`). Set by the deploy harness so the tool image is portable.
- The Caspar node must have `STORAGE_ROOT_PATH` configured (the standard
  Caspar setting); fire VM disks live under `${STORAGE_ROOT_PATH}/vms/`.
- For a real guest boot (vs. scaffold mode), the Caspar node must also have
  `FIRECRACKER_KERNEL_IMAGE` and `FIRECRACKER_ROOTFS_IMAGE` pointing at a
  kernel image and a base rootfs. Without these the controller still
  provisions the persistent disk but skips the guest boot.

## Safety

The sandbox directory is canonicalised and containment-checked on every
`run_vm` and `terminate_vm`: the host refuses to use any path that resolves
outside `${STORAGE_ROOT_PATH}/vms`, so a crafted `vm_id` cannot traverse out
of the sandbox tree. Inside the guest, the only block devices are the
session's own rootfs and data disk — the host filesystem is unreachable.

`risk_level` is `high` because the tool runs arbitrary shell commands; davinci
will route it through its permission engine accordingly.
