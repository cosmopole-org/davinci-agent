"""End-to-end tests for the docker-host bridge gateway client.

A small in-process fake gateway speaks the exact wire protocol so the client's
framing, chunking, correlation and signal delivery are all exercised without a
running node.
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
import json

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, REPO)

from davinci.caspar_bridge import (  # noqa: E402
    CasparBridgeClient, HEADER_LEN, MAX_CHUNK,
    OP_HELLO, OP_WELCOME, OP_REQUEST, OP_RESPONSE, OP_SIGNAL,
)


class FakeGateway:
    """Minimal gateway that reassembles chunked messages and echoes responses."""

    def __init__(self):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.conn = None
        self.hello = None
        self._partials = {}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _recvall(self, sock, n):
        buf = bytearray()
        while len(buf) < n:
            c = sock.recv(n - len(buf))
            if not c:
                return None
            buf.extend(c)
        return bytes(buf)

    def _send(self, opcode, corr, value, msg_id=1):
        payload = json.dumps(value).encode()
        total = max(1, (len(payload) + MAX_CHUNK - 1) // MAX_CHUNK)
        for seq in range(total):
            chunk = payload[seq * MAX_CHUNK:(seq + 1) * MAX_CHUNK]
            body = struct.pack(">BQQII", opcode, msg_id, corr, seq, total) + chunk
            self.conn.sendall(struct.pack(">I", len(body)) + body)

    def _run(self):
        self.conn, _ = self.srv.accept()
        while True:
            hdr = self._recvall(self.conn, 4)
            if hdr is None:
                break
            (flen,) = struct.unpack(">I", hdr)
            body = self._recvall(self.conn, flen)
            if body is None:
                break
            opcode, msg_id, corr, seq, total = struct.unpack(">BQQII", body[:HEADER_LEN])
            chunk = body[HEADER_LEN:]
            part = self._partials.setdefault(msg_id, {})
            part[seq] = chunk
            if len(part) < total:
                continue
            payload = b"".join(part[i] for i in range(total))
            self._partials.pop(msg_id, None)
            value = json.loads(payload) if payload else None
            if opcode == OP_HELLO:
                self.hello = value
                # The node resolves the container's identity from its source IP
                # (here stubbed) and reports it back; the container declares none.
                self._send(OP_WELCOME, corr, {
                    "ok": True, "sessionId": 7,
                    "vmId": "vm1", "machineId": "m1", "programId": "m1", "creatureId": "c1",
                })
            elif opcode == OP_REQUEST:
                # Echo the request back as the result.
                self._send(OP_RESPONSE, corr, {"ok": True, "echo": value})

    def push_signal(self, key, data):
        self._send(OP_SIGNAL, 0, {"key": key, "data": data})


@pytest.fixture()
def gateway():
    gw = FakeGateway()
    yield gw
    try:
        gw.srv.close()
    except OSError:
        pass


def test_handshake_and_call(gateway):
    client = CasparBridgeClient("127.0.0.1", gateway.port, timeout=5)
    client.connect()
    assert client.session_id == 7
    # The container adopts the node-assigned identity from WELCOME.
    assert client.machine_id == "m1" and client.vm_id == "vm1"
    # Wait for the server to record the HELLO.
    for _ in range(50):
        if gateway.hello is not None:
            break
        time.sleep(0.01)
    # The container declares no identity and no secret in the handshake.
    assert gateway.hello == {}

    res = client.call("dbOp", {"op": "get", "key": "k"})
    assert res["ok"] is True
    assert res["echo"] == {"op": "dbOp", "input": {"op": "get", "key": "k"}}
    client.close()


def test_large_payload_chunks(gateway):
    client = CasparBridgeClient("127.0.0.1", gateway.port, timeout=10)
    client.connect()
    big = "x" * (MAX_CHUNK * 2 + 123)  # forces 3 chunks each way
    res = client.call("httpRequest", {"url": "http://e", "body": big})
    assert res["echo"]["input"]["body"] == big
    client.close()


def test_signal_delivery(gateway):
    received = []
    client = CasparBridgeClient("127.0.0.1", gateway.port, timeout=5)
    client.on_signal(lambda key, data: received.append((key, data)))
    client.connect()
    gateway.push_signal("creatures/signal", {"kind": "invoke", "correlationId": "abc"})
    for _ in range(100):
        if received:
            break
        time.sleep(0.01)
    assert received and received[0][0] == "creatures/signal"
    assert received[0][1]["correlationId"] == "abc"
    client.close()


def test_signaling_client_unwraps_correlated_signal_shapes():
    from davinci.caspar_signaling import _find_correlated_signal

    direct = {"kind": "tools/result", "correlationId": "cid", "result": {"ok": True}}
    wrapped = {"key": "creatures/signal", "data": json.dumps(direct)}
    stores_wrapped = {"data": json.dumps({"data": json.dumps(direct)})}

    assert _find_correlated_signal(direct, "cid") == direct
    assert _find_correlated_signal(wrapped, "cid") == direct
    assert _find_correlated_signal(stores_wrapped, "cid") == direct
    assert _find_correlated_signal(wrapped, "other") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
