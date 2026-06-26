"""Self-contained Caspar signalling client used by Davinci.

This is the single client both the host-side deploy/test harness and the
Davinci docker-creature use to talk to a Caspar node over its binary TCP action
protocol. It implements:

* the wire framing (no tag byte on requests; buffer+ACK flow control),
* RSA-PSS request signing (the scheme the node verifies for external callers),
* convenience methods for the creature lifecycle and signalling:
  ``login``, ``create_machine_creature``, ``create_program``, ``deploy``,
  ``run_entity``, ``read_vm_logs``, ``signal``.

RSA signing needs ``pycryptodome``; everything else is stdlib. If pycryptodome
is unavailable the client still works for unsigned/insider flows.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:  # signing is only needed for authenticated external calls
    from Crypto.PublicKey import RSA
    from Crypto.Signature import pss
    from Crypto.Hash import SHA256
    _HAVE_CRYPTO = True
except Exception:  # pragma: no cover - exercised only without the dep
    _HAVE_CRYPTO = False


def _lp(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack(">I", len(b)) + b


def sign_payload(priv_pem: str, payload_bytes: bytes) -> str:
    """RSA-PSS SHA256 signature, base64 — matches the node's verifier."""
    if not _HAVE_CRYPTO:
        raise RuntimeError("pycryptodome is required to sign Caspar requests")
    key = RSA.import_key(priv_pem)
    h = SHA256.new(payload_bytes)
    return base64.b64encode(pss.new(key).sign(h)).decode()


class CasparSignalingClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8074, timeout: float = 60.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.user_id: str = ""
        self.priv_pem: str = ""

    # -- connection ----------------------------------------------------------
    def connect(self) -> "CasparSignalingClient":
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        return self

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self) -> "CasparSignalingClient":
        return self.connect()

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- low-level request ---------------------------------------------------
    def _recvall(self, n: int) -> bytes:
        assert self.sock is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed while reading")
            buf.extend(chunk)
        return bytes(buf)

    def send(self, path: str, payload: Dict[str, Any], *, sign: bool = True) -> Dict[str, Any]:
        if self.sock is None:
            raise RuntimeError("not connected")
        payload_bytes = json.dumps(payload).encode("utf-8")
        signature = ""
        if sign and self.priv_pem:
            signature = sign_payload(self.priv_pem, payload_bytes)
        pkt_id = str(uuid.uuid4())
        body = _lp(signature) + _lp(self.user_id) + _lp(path) + _lp(pkt_id) + payload_bytes
        self.sock.sendall(struct.pack(">I", len(body)) + body)

        while True:
            hdr = self._recvall(4)
            length = struct.unpack(">I", hdr)[0]
            if length == 0:
                return {}
            frame = self._recvall(length)
            tag = frame[0]
            if tag == 0x02:  # response
                off = 1
                pl = struct.unpack(">I", frame[off:off + 4])[0]; off += 4
                off += pl  # skip packet id
                res_code = struct.unpack(">I", frame[off:off + 4])[0]; off += 4
                payload_out = frame[off:]
                self.sock.sendall(struct.pack(">I", 1) + b"\x01")  # ACK
                result: Dict[str, Any] = {}
                if payload_out:
                    try:
                        result = json.loads(payload_out)
                    except json.JSONDecodeError:
                        result = {"raw": payload_out.decode("utf-8", "replace")}
                result.setdefault("_res_code", res_code)
                return result
            # update/signal frames are fire-and-forget — keep reading
            continue

    # -- high-level creature lifecycle --------------------------------------
    def login(self, username: str) -> Dict[str, Any]:
        r = self.send("/creatures/login", {"username": username,
                                           "emailToken": f"{username}@dev.local",
                                           "metadata": {}}, sign=False)
        if r.get("_res_code", -1) != 0:
            raise RuntimeError(f"login failed: {r}")
        self.user_id = r["user"]["id"]
        self.priv_pem = r["privateKey"]
        return r

    def create_machine_creature(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        r = self.send("/creatures/create", {"type": "machine", "username": name[:32],
                                            "publicKey": "", "metadata": metadata or {}})
        if r.get("_res_code", -1) != 0:
            raise RuntimeError(f"create machine creature failed: {r}")
        return r["creature"]["id"]

    def create_program(self, app_id: str, path: str, runtime: str, comment: str = "") -> str:
        r = self.send("/programs/create", {"appId": app_id, "path": path,
                                           "Comment": comment, "runtime": runtime, "publicKey": ""})
        if r.get("_res_code", -1) != 0:
            raise RuntimeError(f"create program failed: {r}")
        return r.get("program", {}).get("id", "")

    def deploy(self, program_id: str, entity_id: str, entity_type: str,
               primary_b64: str, files_b64: Optional[Dict[str, str]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = dict(metadata or {})
        if files_b64:
            meta["files"] = files_b64
        r = self.send("/programs/deploy", {"machineId": program_id, "entityId": entity_id,
                                           "entityType": entity_type, "downloadable": False,
                                           "payload": primary_b64, "metadata": meta})
        if r.get("_res_code", -1) != 0:
            raise RuntimeError(f"deploy failed: {r}")
        return r

    def run_entity_with_attachments(
        self, program_id: str, entity_id: str, *,
        objective: str,
        attachments: Optional[List[Any]] = None,
        task_extra: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        ram_mb: int = 512, disk_gb: int = 1, cpu_cores: int = 1,
        max_exec_seconds: int = 240,
    ) -> str:
        """Signal a Davinci creature with a multimodal prompt.

        ``attachments`` is a list of either local file paths (the bytes are read
        and base64-encoded inline) or dicts conforming to the ``task.json``
        attachment spec (``{name, mime_type, data | path, description}``). The
        composed payload is delivered to ``/app/input/task.json``; the davinci
        creature materialises each inline attachment to
        ``/app/input/attachments/<name>`` before invoking the agent loop.
        """
        from .attachments import attachment_from_file
        attach_specs: List[Dict[str, Any]] = []
        for entry in attachments or []:
            if isinstance(entry, str):
                attach_specs.append(attachment_from_file(entry))
            elif isinstance(entry, dict):
                attach_specs.append(entry)
            else:
                raise TypeError(f"unsupported attachment: {entry!r}")
        task: Dict[str, Any] = {"objective": objective}
        if attach_specs:
            task["attachments"] = attach_specs
        if task_extra:
            task.update(task_extra)
        params: Dict[str, str] = {"task.json": json.dumps(task)}
        if config:
            params["config.json"] = json.dumps(config)
        return self.run_entity(program_id, entity_id, params=params,
                               ram_mb=ram_mb, disk_gb=disk_gb,
                               cpu_cores=cpu_cores,
                               max_exec_seconds=max_exec_seconds)

    def run_entity(self, program_id: str, entity_id: str, *, params: Optional[Dict[str, str]] = None,
                   ram_mb: int = 512, disk_gb: int = 1, cpu_cores: int = 1,
                   max_exec_seconds: int = 120) -> str:
        r = self.send("/programs/runEntity", {
            "programId": program_id, "machineId": program_id, "entityId": entity_id,
            "resources": {"ramMb": ram_mb, "diskGb": disk_gb, "cpuCores": cpu_cores,
                          "maxExecTimeSeconds": max_exec_seconds},
            "params": params or {},
        })
        if r.get("_res_code", -1) != 0:
            raise RuntimeError(f"runEntity failed: {r}")
        return r.get("vmId", "")

    def read_vm_logs(self, vm_id: str, log_type: str = "", count: int = 500, offset: int = 0) -> List[Any]:
        r = self.send("/machines/readVmLogs", {"vmId": vm_id, "logType": log_type,
                                               "count": count, "offset": offset})
        return r.get("logs", []) if isinstance(r, dict) else []

    def signal(self, *, creature_type: str = "machine", data: Any = None,
               creature_id: str = "", program_id: str = "", entity_id: str = "") -> Dict[str, Any]:
        return self.send("/creatures/signal", {
            "type": creature_type,
            "data": data if isinstance(data, str) else json.dumps(data or {}),
            "creatureId": creature_id, "programId": program_id, "entityId": entity_id,
        })

    # -- miniapp request/response over signalling ----------------------------
    def signal_miniapp(self, *, creature_id: str, program_id: str, action: str,
                       payload: Dict[str, Any], entity: str = "main",
                       store_id: str = "", timeout: float = 30.0) -> Dict[str, Any]:
        """Call a miniapp creature and await its result.

        Mirrors the Decillion CLI's ``signalMiniapp``: ship a ``/creatures/signal``
        carrying ``{action, programId, entity, correlationId, payload}`` and then
        capture the ``creatures/signal/result`` *update* frame (tag ``0x01``) the
        creature emits, matched by ``correlationId``.
        """
        correlation_id = uuid.uuid4().hex
        # The miniapp creature's ``unwrapSignal`` parses a *doubly*-encoded
        # envelope (matching the Decillion CLI / bench client exactly):
        #   data    = {"correlationId": cid, "payload": <inner JSON string>}
        #   inner   = {"action": <action>, "payload": {...args..., correlationId}}
        # ``inner.action`` becomes the dispatched path and ``inner.payload`` the
        # action args. Encoding ``action``/``payload`` at the wrong level (a raw
        # dict, or ``action`` at the data layer) makes the creature read an empty
        # path and silently no-op, so we mirror the working client byte-for-byte.
        inner_payload = json.dumps({
            "action": action,
            "payload": {**(payload or {}), "correlationId": correlation_id},
        })
        data = json.dumps({"correlationId": correlation_id, "payload": inner_payload})
        # "pvp" is the node's program-targeted signal type: it delivers the
        # packet to the target program (creatureId/programId) which runs its
        # entity and signals the result back to us.
        req = {"type": "pvp", "data": data, "creatureId": creature_id,
               "programId": program_id, "entityId": entity, "storeId": store_id,
               "temp": False}
        return self._signal_await_result("/creatures/signal", req, correlation_id, timeout)

    def signal_entity_await(self, *, creature_id: str, program_id: str, entity_id: str,
                            envelope: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
        """Signal a *running* creature VM (docker tool or the Davinci agent) and
        await its correlated reply — purely over the signaling API.

        This is how an external client hands work to a standalone creature: a
        ``/creatures/signal`` (pvp) carrying ``envelope`` is delivered by the node
        as a pushed signal onto the creature's live gateway connection; the
        creature processes it and signals a reply back to *this* client (its
        ``user_id``), matched by ``correlationId``. The creature is never
        cold-spawned per call and never reads a task file — it must already be
        running (started once via ``run_entity``) and serving signals.
        """
        correlation_id = uuid.uuid4().hex
        data = json.dumps({**envelope, "correlationId": correlation_id, "reply_to": self.user_id})
        req = {"type": "pvp", "data": data, "creatureId": creature_id,
               "programId": program_id, "entityId": entity_id, "temp": False}
        return self._signal_await_result("/creatures/signal", req, correlation_id, timeout)

    def _signal_await_result(self, path: str, payload: Dict[str, Any],
                             correlation_id: str, timeout: float) -> Dict[str, Any]:
        if self.sock is None:
            raise RuntimeError("not connected")
        payload_bytes = json.dumps(payload).encode("utf-8")
        signature = sign_payload(self.priv_pem, payload_bytes) if self.priv_pem else ""
        pkt_id = str(uuid.uuid4())
        body = _lp(signature) + _lp(self.user_id) + _lp(path) + _lp(pkt_id) + payload_bytes
        self.sock.sendall(struct.pack(">I", len(body)) + body)

        deadline = time.time() + timeout
        ack: Dict[str, Any] = {}
        old_to = self.sock.gettimeout()
        try:
            while time.time() < deadline:
                self.sock.settimeout(max(0.5, deadline - time.time()))
                try:
                    hdr = self._recvall(4)
                except (socket.timeout, ConnectionError):
                    break
                length = struct.unpack(">I", hdr)[0]
                if length == 0:
                    continue
                frame = self._recvall(length)
                tag = frame[0]
                if tag == 0x02:  # response ack to our /creatures/signal
                    off = 1
                    pl = struct.unpack(">I", frame[off:off + 4])[0]; off += 4
                    off += pl
                    res_code = struct.unpack(">I", frame[off:off + 4])[0]; off += 4
                    self.sock.sendall(struct.pack(">I", 1) + b"\x01")  # ACK
                    try:
                        ack = json.loads(frame[off:]) if frame[off:] else {}
                    except json.JSONDecodeError:
                        ack = {}
                    ack["_res_code"] = res_code
                    if res_code != 0:
                        return {"ok": False, "error": "signal rejected", "ack": ack}
                    continue
                if tag == 0x01:  # update / pushed signal frame: [0x01][key][payload]
                    off = 1
                    klen = struct.unpack(">I", frame[off:off + 4])[0]; off += 4
                    off += klen  # skip key
                    try:
                        msg = json.loads(frame[off:])
                    except json.JSONDecodeError:
                        continue
                    matched = _find_correlated_signal(msg, correlation_id)
                    if matched is not None:
                        return {"ok": True, "result": matched}
                    continue
                # other tags: ignore and keep reading
        finally:
            try:
                self.sock.settimeout(old_to)
            except OSError:
                pass
        return {"ok": False, "error": "miniapp signal result timeout",
                "correlationId": correlation_id, "ack": ack}

    # -- helpers -------------------------------------------------------------
    def wait_for_vm_log(self, vm_id: str, marker: str, *, timeout: float = 90.0,
                        poll: float = 2.0) -> Tuple[bool, List[Any]]:
        """Poll VM logs until a line containing ``marker`` appears or timeout."""
        deadline = time.time() + timeout
        logs: List[Any] = []
        while time.time() < deadline:
            logs = self.read_vm_logs(vm_id)
            if any(marker in _log_text(l) for l in logs):
                return True, logs
            time.sleep(poll)
        return False, logs



def _maybe_json(value: Any) -> Any:
    """Decode JSON strings used by Caspar signal wrappers, if present."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _find_correlated_signal(message: Any, correlation_id: str) -> Optional[Dict[str, Any]]:
    """Return the signal payload matching ``correlation_id`` across node shapes.

    Caspar has emitted pushed signal updates in a few compatible envelopes over
    time: sometimes the update payload *is* the caller's packet, sometimes it is
    wrapped as ``{key, data}``, and program-targeted signals can add a second
    ``data`` JSON string inside a StoresSend-style object.  The crypto E2E smoke
    test only cares about the correlated reply packet, so unwrap these transport
    layers before deciding whether no result arrived.
    """
    seen: set[int] = set()

    def walk(value: Any) -> Optional[Dict[str, Any]]:
        value = _maybe_json(value)
        if not isinstance(value, dict):
            return None
        ident = id(value)
        if ident in seen:
            return None
        seen.add(ident)
        if value.get("correlationId") == correlation_id:
            return value
        # Common pushed-signal wrappers: external socket updates may expose the
        # signal value under ``data``; StoresSend-style payloads may then encode
        # the real packet as another JSON string in that same field.
        for key in ("data", "packet", "payload", "result"):
            if key in value:
                found = walk(value.get(key))
                if found is not None:
                    return found
        return None

    return walk(message)

def _log_text(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        # VM-log rows serialize the line under "data"; fall back to other shapes.
        return entry.get("data") or entry.get("text") or entry.get("line") or json.dumps(entry)
    return str(entry)
