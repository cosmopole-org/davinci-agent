"""Docker-host bridge gateway client.

When Davinci (or any tool) runs as a Caspar ``docker`` creature it is sandboxed
on the gateway docker network with **no other route to the outside world**. Its
only channel is a single long-lived TCP connection to the node's *docker-host
bridge gateway*. Over that one connection the container performs *every*
interaction with the outside world — persisting through DB/storage ops, making
outbound HTTP calls, signalling sibling creatures — and receives pushed signals
from other creatures.

This module implements the gateway's **chunked wire protocol** and a small,
thread-safe client:

* :class:`CasparBridgeClient` — connect / handshake / call / receive-signals.
* :func:`bridge_from_env` — construct a client from the ``CASPAR_GATEWAY_*`` and
  ``CASPAR_VM_ID``/``CASPAR_PROGRAM_ID`` env vars the node injects at container
  start.

Wire format (mirrors the Rust node, ``drivers/vmm/network/docker_host``)::

    [u32 BE frame_len][frame body]
    frame body = [u8 op][u64 BE msg_id][u64 BE corr_id][u32 BE seq][u32 BE total][chunk]

A logical message larger than ``MAX_CHUNK`` is split across several frames that
share ``msg_id`` and are reassembled by the receiver.

The module is stdlib-only.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
from typing import Any, Callable, Dict, Optional

# ── Opcodes (must match the node) ──────────────────────────────────────────────
OP_HELLO = 0x01
OP_WELCOME = 0x02
OP_REQUEST = 0x10
OP_RESPONSE = 0x11
OP_SIGNAL = 0x20
OP_PING = 0x30
OP_PONG = 0x31
OP_ERROR = 0x40

HEADER_LEN = 1 + 8 + 8 + 4 + 4  # op + msg_id + corr_id + seq + total
MAX_CHUNK = 64 * 1024
MAX_FRAME = 32 * 1024 * 1024


class BridgeError(RuntimeError):
    """Raised on handshake/transport/protocol failures."""


SignalHandler = Callable[[str, Any], None]


class CasparBridgeClient:
    """Thread-safe client for the docker-host bridge gateway.

    A single background thread reads frames, reassembles messages, routes
    ``RESPONSE`` frames to the waiting caller (matched by correlation id) and
    delivers ``SIGNAL`` frames to a registered handler.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        token: str = "",
        timeout: float = 60.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        # The opaque, node-issued session token is the ONLY credential we hold.
        # The container never declares its own identity; the node resolves it
        # from the token and reports it back in the WELCOME handshake.
        self.token = token
        self.vm_id = ""
        self.machine_id = ""
        self.program_id = ""
        self.creature_id = ""
        self.timeout = timeout

        self._sock: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._next_id_lock = threading.Lock()
        self._next_id = 1
        self._pending: Dict[int, threading.Event] = {}
        self._results: Dict[int, Any] = {}
        self._pending_lock = threading.Lock()
        self._partials: Dict[int, Dict[str, Any]] = {}
        self._signal_handler: Optional[SignalHandler] = None
        self._reader: Optional[threading.Thread] = None
        self._closed = threading.Event()
        self.session_id: Optional[int] = None

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> "CasparBridgeClient":
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._reader = threading.Thread(target=self._read_loop, name="caspar-bridge-reader", daemon=True)
        self._reader.start()
        self._handshake()
        return self

    def _handshake(self) -> None:
        corr = self._alloc_id()
        ev = self._register(corr)
        self._send(OP_HELLO, corr, {"token": self.token})
        if not ev.wait(self.timeout):
            raise BridgeError("handshake timed out waiting for WELCOME")
        res = self._take(corr)
        if not isinstance(res, dict) or not res.get("ok", False):
            raise BridgeError(f"handshake rejected: {res}")
        self.session_id = res.get("sessionId")
        # Adopt the node-assigned (authoritative) identity reported back to us.
        self.vm_id = res.get("vmId", "")
        self.machine_id = res.get("machineId", "")
        self.program_id = res.get("programId", "")
        self.creature_id = res.get("creatureId", "")

    def close(self) -> None:
        self._closed.set()
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def __enter__(self) -> "CasparBridgeClient":
        return self.connect()

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- signal handling ----------------------------------------------------
    def on_signal(self, handler: SignalHandler) -> None:
        """Register a callback invoked (on the reader thread) for each pushed
        signal as ``handler(key, data)``."""
        self._signal_handler = handler

    # -- host calls ---------------------------------------------------------
    def call(self, op: str, input: Optional[Dict[str, Any]] = None, *, timeout: Optional[float] = None) -> Any:
        """Invoke a node host function and return its parsed JSON result.

        ``op`` is any of the node's unified host-function names (``dbOp``,
        ``httpRequest``, ``signal``, ``signalUser``, ``runVm``, ...). The node
        stamps the connection's verified identity onto the request, so callers
        never need to pass ``vmId``/``programId`` themselves.
        """
        corr = self._alloc_id()
        ev = self._register(corr)
        self._send(OP_REQUEST, corr, {"op": op, "input": input or {}})
        if not ev.wait(timeout if timeout is not None else self.timeout):
            self._take(corr)
            raise BridgeError(f"host call '{op}' timed out")
        return self._take(corr)

    # -- high-level convenience wrappers -----------------------------------
    def db_put(self, key: str, val: str, **extra: Any) -> Any:
        return self.call("dbOp", {"op": "put", "key": key, "val": val, **extra})

    def db_get(self, key: str, **extra: Any) -> Any:
        return self.call("dbOp", {"op": "get", "key": key, **extra})

    def db_del(self, key: str, **extra: Any) -> Any:
        return self.call("dbOp", {"op": "del", "key": key, **extra})

    def http_request(self, url: str, *, method: str = "GET", headers: Optional[Dict[str, str]] = None,
                     body: Optional[str] = None, timeout: Optional[float] = None) -> Any:
        """Make an outbound HTTP request *through the node*. Returns the node's
        result; the body is base64-encoded under the result's data field."""
        payload: Dict[str, Any] = {"url": url, "method": method}
        if headers:
            payload["headers"] = headers
        if body is not None:
            payload["body"] = body
        return self.call("httpRequest", payload, timeout=timeout)

    def signal_user(self, key: str, user_id: str, packet: Any) -> Any:
        return self.call("signalUser", {"key": key, "userId": user_id,
                                        "packet": packet if isinstance(packet, str) else json.dumps(packet)})

    def signal_group(self, key: str, group_id: str, packet: Any, except_ids: Optional[list] = None) -> Any:
        return self.call("signalGroup", {"key": key, "groupId": group_id,
                                         "packet": packet if isinstance(packet, str) else json.dumps(packet),
                                         "except": except_ids or []})

    def ping(self, *, timeout: Optional[float] = None) -> Any:
        corr = self._alloc_id()
        ev = self._register(corr)
        self._send(OP_PING, corr, {})
        if not ev.wait(timeout if timeout is not None else self.timeout):
            self._take(corr)
            raise BridgeError("ping timed out")
        return self._take(corr)

    # -- internals ----------------------------------------------------------
    def _alloc_id(self) -> int:
        with self._next_id_lock:
            v = self._next_id
            self._next_id += 1
            return v

    def _register(self, corr: int) -> threading.Event:
        ev = threading.Event()
        with self._pending_lock:
            self._pending[corr] = ev
        return ev

    def _take(self, corr: int) -> Any:
        with self._pending_lock:
            self._pending.pop(corr, None)
            return self._results.pop(corr, None)

    def _complete(self, corr: int, result: Any) -> None:
        with self._pending_lock:
            ev = self._pending.get(corr)
            if ev is not None:
                self._results[corr] = result
                ev.set()

    def _send(self, opcode: int, corr: int, value: Any) -> None:
        if self._sock is None:
            raise BridgeError("not connected")
        payload = json.dumps(value).encode("utf-8") if value is not None else b""
        msg_id = self._alloc_id()
        total = max(1, (len(payload) + MAX_CHUNK - 1) // MAX_CHUNK)
        with self._send_lock:
            for seq in range(total):
                chunk = payload[seq * MAX_CHUNK:(seq + 1) * MAX_CHUNK]
                body = struct.pack(">BQQII", opcode, msg_id, corr, seq, total) + chunk
                frame = struct.pack(">I", len(body)) + body
                self._sock.sendall(frame)

    def _recvall(self, n: int) -> Optional[bytes]:
        sock = self._sock
        if sock is None:
            return None
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _read_loop(self) -> None:
        while not self._closed.is_set():
            hdr = self._recvall(4)
            if hdr is None:
                break
            (frame_len,) = struct.unpack(">I", hdr)
            if frame_len == 0:
                continue
            if frame_len > MAX_FRAME or frame_len < HEADER_LEN:
                break
            body = self._recvall(frame_len)
            if body is None:
                break
            self._dispatch_frame(body)
        # Wake any waiters so they fail fast instead of hanging.
        with self._pending_lock:
            for corr, ev in list(self._pending.items()):
                self._results.setdefault(corr, None)
                ev.set()

    def _dispatch_frame(self, body: bytes) -> None:
        opcode, msg_id, corr, seq, total = struct.unpack(">BQQII", body[:HEADER_LEN])
        chunk = body[HEADER_LEN:]
        if total <= 1:
            self._on_message(opcode, corr, chunk)
            return
        part = self._partials.setdefault(msg_id, {"total": total, "chunks": {}, "op": opcode, "corr": corr})
        part["chunks"][seq] = chunk
        if len(part["chunks"]) < total:
            return
        self._partials.pop(msg_id, None)
        payload = b"".join(part["chunks"][i] for i in range(total))
        self._on_message(opcode, corr, payload)

    def _on_message(self, opcode: int, corr: int, payload: bytes) -> None:
        value: Any = None
        if payload:
            try:
                value = json.loads(payload)
            except json.JSONDecodeError:
                value = {"raw": payload.decode("utf-8", "replace")}
        if opcode in (OP_RESPONSE, OP_WELCOME, OP_PONG, OP_ERROR):
            self._complete(corr, value)
        elif opcode == OP_SIGNAL:
            handler = self._signal_handler
            if handler is not None and isinstance(value, dict):
                try:
                    handler(value.get("key", ""), value.get("data"))
                except Exception:  # noqa: BLE001 — a bad handler must not kill the reader
                    pass


def bridge_from_env(*, connect: bool = True, timeout: float = 60.0) -> Optional[CasparBridgeClient]:
    """Build a client from the env the node injects at container start.

    The node provides only the gateway address and an opaque
    ``CASPAR_SESSION_TOKEN``; the container's identity is resolved server-side
    from that token. Returns ``None`` when ``CASPAR_GATEWAY_HOST`` is unset
    (i.e. not running as a gateway-managed docker creature), so callers can fall
    back gracefully.
    """
    host = os.environ.get("CASPAR_GATEWAY_HOST", "").strip()
    if not host:
        return None
    port = int(os.environ.get("CASPAR_GATEWAY_PORT", "8079") or "8079")
    token = os.environ.get("CASPAR_SESSION_TOKEN", "").strip()
    client = CasparBridgeClient(host, port, token=token, timeout=timeout)
    if connect:
        client.connect()
    return client
