"""Unit tests for Davinci's multimodal attachment plumbing."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davinci import (  # noqa: E402
    Attachment, DavinciAgent, INLINE_MAX_BYTES,
    attachment_from_file, materialize_attachments,
)
from davinci.gemini_reasoner import GeminiReasoner  # noqa: E402


PNG_1x1 = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
           b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc````"
           b"\x00\x00\x00\x05\x00\x01]\xcc\xdb\x8f\x00\x00\x00\x00IEND\xaeB`\x82")


def test_attachment_from_file_roundtrips_bytes():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "pic.png")
        with open(src, "wb") as fh:
            fh.write(PNG_1x1)
        spec = attachment_from_file(src)
    assert spec["name"] == "pic.png"
    assert spec["mime_type"] == "image/png"
    assert base64.b64decode(spec["data"]) == PNG_1x1


def test_materialize_inline_data_writes_file_to_disk():
    spec = {"attachments": [{
        "name": "pic.png",
        "mime_type": "image/png",
        "data": base64.b64encode(PNG_1x1).decode(),
    }]}
    with tempfile.TemporaryDirectory() as tmp:
        atts = materialize_attachments(spec, tmp)
        assert len(atts) == 1
        att = atts[0]
        assert att.name == "pic.png"
        assert att.mime_type == "image/png"
        assert att.size == len(PNG_1x1)
        assert att.kind == "image"
        assert att.path == os.path.join(tmp, "attachments", "pic.png")
        assert att.read_bytes() == PNG_1x1


def test_materialize_accepts_data_url_prefix():
    spec = {"attachments": [{
        "name": "tiny.png",
        "data": "data:image/png;base64," + base64.b64encode(PNG_1x1).decode(),
    }]}
    with tempfile.TemporaryDirectory() as tmp:
        atts = materialize_attachments(spec, tmp)
    assert atts[0].mime_type == "image/png"
    assert atts[0].size == len(PNG_1x1)


def test_materialize_resolves_already_uploaded_path():
    with tempfile.TemporaryDirectory() as tmp:
        rel = "attachments/already.png"
        full = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as fh:
            fh.write(PNG_1x1)
        spec = {"attachments": [{"name": "already.png", "path": rel,
                                  "mime_type": "image/png"}]}
        atts = materialize_attachments(spec, tmp)
    assert len(atts) == 1
    assert atts[0].path == full
    assert atts[0].size == len(PNG_1x1)


def test_materialize_rejects_path_traversal_and_unsafe_names():
    with tempfile.TemporaryDirectory() as tmp:
        # Names with path separators are flattened to a safe basename so the
        # file always lands under <input>/attachments/.
        spec = {"attachments": [{"name": "../escape.png",
                                  "data": base64.b64encode(PNG_1x1).decode()}]}
        atts = materialize_attachments(spec, tmp)
        assert len(atts) == 1
        assert atts[0].name == "escape.png"
        assert atts[0].path == os.path.join(tmp, "attachments", "escape.png")

        # Names that don't survive the basename whitelist are rejected outright.
        spec = {"attachments": [{"name": " ",
                                  "data": base64.b64encode(PNG_1x1).decode()}]}
        assert materialize_attachments(spec, tmp) == []

        # A path that resolves outside the input dir is rejected.
        outside = os.path.abspath(os.path.join(tmp, "..", "evil.png"))
        os.makedirs(os.path.dirname(outside), exist_ok=True)
        try:
            with open(outside, "wb") as fh:
                fh.write(PNG_1x1)
            spec = {"attachments": [{"name": "ok.png", "path": "../evil.png"}]}
            assert materialize_attachments(spec, tmp) == []
        finally:
            try:
                os.unlink(outside)
            except OSError:
                pass


def test_run_passes_attachments_to_working_memory():
    agent = DavinciAgent()
    att = Attachment(name="x.png", mime_type="image/png",
                     path="/tmp/nonexistent.png", size=42)
    result = agent.run("self-test the multimodal flow",
                       required_categories=["analysis", "synthesis"],
                       attachments=[att])
    assert result.success
    stored = agent.working.get_fact("attachments")
    assert stored == [att]
    # the run_start trace event records the attachment summary
    starts = [e for e in agent.tracer.events if e.kind == "run_start"]
    assert starts and starts[0].data["attachments"][0]["name"] == "x.png"


def test_gemini_reasoner_inlines_small_attachments():
    reasoner = GeminiReasoner("fake-key")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tiny.png")
        with open(path, "wb") as fh:
            fh.write(PNG_1x1)
        att = Attachment(name="tiny.png", mime_type="image/png",
                         path=path, size=len(PNG_1x1))
        part = reasoner._attachment_part(att)
    assert part is not None
    assert "inlineData" in part
    assert part["inlineData"]["mimeType"] == "image/png"
    assert base64.b64decode(part["inlineData"]["data"]) == PNG_1x1


def test_gemini_reasoner_generate_packs_attachments_into_parts():
    """Real LLM not called — we mock urlopen and assert the body shape."""
    reasoner = GeminiReasoner("fake-key", max_retries=1, timeout=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tiny.png")
        with open(path, "wb") as fh:
            fh.write(PNG_1x1)
        att = Attachment(name="tiny.png", mime_type="image/png",
                         path=path, size=len(PNG_1x1))

        captured = {}

        class _FakeResp(io.BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            headers = {}

        def _fake_urlopen(req, *a, **kw):
            # Capture the request body so we can introspect the parts array.
            captured["body"] = req.data
            payload = json.dumps({
                "candidates": [{
                    "content": {"parts": [{"text": "{\"tool\": null, "
                                          "\"thought\": \"ok\", "
                                          "\"final_answer\": \"saw image\"}"}]}
                }]
            }).encode()
            return _FakeResp(payload)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            text = reasoner._generate("describe the image",
                                       attachments=[att])
        assert text is not None
        sent = json.loads(captured["body"])
        parts = sent["contents"][0]["parts"]
        # First the inlineData part, then the text prompt.
        assert "inlineData" in parts[0]
        assert parts[0]["inlineData"]["mimeType"] == "image/png"
        assert parts[-1]["text"] == "describe the image"
