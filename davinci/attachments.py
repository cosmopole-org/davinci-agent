"""Multimodal attachment handling for Davinci.

The signalling API lets a caller ship arbitrary files alongside the prompt:

* inline, in ``task.json`` under ``attachments``:
  ``{"name": "photo.png", "mime_type": "image/png", "data": "<base64>"}``
* or already-uploaded under ``params`` and referenced by relative path:
  ``{"name": "photo.png", "mime_type": "image/png", "path": "attachments/photo.png"}``

This module:

1. parses + normalises the attachment spec from ``task.json``,
2. **writes each attachment to the creature's filesystem** under
   ``$DAVINCI_INPUT_DIR/attachments/<name>`` so the agent (and any tool it
   delegates to) can read it as a real file, and
3. exposes a stable :class:`Attachment` dataclass the reasoner consumes to
   build standard multimodal LLM requests (Gemini ``inlineData`` /
   ``fileData``, Claude/OpenAI ``image_url`` etc.).

Everything is stdlib-only so it runs inside the slim creature container.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Gemini's inline-bytes limit is 20 MiB per part; above this we fall back to
# the Files API (see :class:`davinci.llm.GeminiClient`). Keep this conservative.
INLINE_MAX_BYTES = 18 * 1024 * 1024

# Names a creature is allowed to write under its attachments dir. Reject
# anything that escapes the directory or contains a path separator. The check
# is intentionally strict because callers control the filename.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._-]{0,127}$")


@dataclass
class Attachment:
    """A multimodal attachment materialised on the creature's filesystem."""

    name: str
    mime_type: str
    path: str                       # absolute path inside the container
    size: int                       # bytes
    description: str = ""           # optional caller-supplied caption
    role: str = "user"              # "user" by convention; reserved for future
    extra: Dict[str, Any] = field(default_factory=dict)

    def read_bytes(self) -> bytes:
        with open(self.path, "rb") as fh:
            return fh.read()

    def read_base64(self) -> str:
        return base64.b64encode(self.read_bytes()).decode("ascii")

    @property
    def kind(self) -> str:
        """Coarse media kind used by reasoners that don't accept arbitrary MIME."""
        major = (self.mime_type or "").split("/", 1)[0].lower()
        if major in ("image", "audio", "video"):
            return major
        if self.mime_type == "application/pdf":
            return "document"
        if major == "text" or self.mime_type in ("application/json",
                                                 "application/xml"):
            return "text"
        return "binary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mime_type": self.mime_type,
            "path": self.path,
            "size": self.size,
            "description": self.description,
            "kind": self.kind,
        }


# --------------------------------------------------------------------------- #
# Resolution: task.json → on-disk Attachment objects
# --------------------------------------------------------------------------- #

def _guess_mime(name: str, fallback: str = "application/octet-stream") -> str:
    mt, _ = mimetypes.guess_type(name)
    return mt or fallback


def _safe_name(name: str) -> str:
    """Reject path-traversal / unsafe names; return a clean basename."""
    name = os.path.basename(str(name or "").strip())
    if not name or not _SAFE_NAME.match(name):
        raise ValueError(f"unsafe attachment name: {name!r}")
    return name


def _decode_data_url_or_b64(data: str) -> Tuple[bytes, Optional[str]]:
    """Accept ``data:<mime>;base64,<payload>`` or a bare base64 string.

    Returns ``(bytes, mime_hint or None)``.
    """
    s = str(data or "").strip()
    if s.startswith("data:"):
        head, _, body = s.partition(",")
        mime_hint = head[5:].split(";", 1)[0] or None
        return base64.b64decode(body, validate=False), mime_hint
    # tolerate whitespace / newlines that JSON encoders sometimes inject
    return base64.b64decode("".join(s.split()), validate=False), None


def materialize_attachments(
    spec: Any, input_dir: str,
) -> List[Attachment]:
    """Write any inline-data attachments to disk and return :class:`Attachment`s.

    ``spec`` is the raw value of ``task.json``'s ``attachments`` field (a list)
    or the task dict itself; anything else returns ``[]``.

    Inline-data attachments are decoded once and written under
    ``<input_dir>/attachments/<name>``. Already-uploaded attachments (``path``)
    are resolved against ``input_dir`` and validated.
    """
    if isinstance(spec, dict):
        spec = spec.get("attachments")
    if not isinstance(spec, list) or not spec:
        return []

    out_dir = os.path.join(input_dir, "attachments")
    os.makedirs(out_dir, exist_ok=True)

    resolved: List[Attachment] = []
    for entry in spec:
        if not isinstance(entry, dict):
            continue
        try:
            attachment = _materialize_one(entry, input_dir, out_dir)
        except Exception as exc:  # noqa: BLE001 — surface, then skip
            print("DAVINCI_ATTACHMENT " + json.dumps(
                {"error": repr(exc)[:200], "entry": _summary(entry)}), flush=True)
            continue
        resolved.append(attachment)
    return resolved


def _materialize_one(entry: Dict[str, Any], input_dir: str,
                     out_dir: str) -> Attachment:
    name = _safe_name(entry.get("name") or entry.get("filename") or "")
    description = str(entry.get("description") or entry.get("caption") or "")
    mime = str(entry.get("mime_type") or entry.get("mimeType") or "").strip()

    if "data" in entry and entry["data"] is not None:
        raw, mime_hint = _decode_data_url_or_b64(entry["data"])
        target = os.path.join(out_dir, name)
        with open(target, "wb") as fh:
            fh.write(raw)
        size = len(raw)
        if not mime:
            mime = mime_hint or _guess_mime(name)
    else:
        rel = str(entry.get("path") or "").strip()
        if not rel:
            raise ValueError("attachment requires either 'data' or 'path'")
        # Resolve relative to input_dir and refuse anything outside it.
        candidate = os.path.abspath(os.path.join(input_dir, rel))
        root = os.path.abspath(input_dir)
        if not candidate.startswith(root + os.sep) and candidate != root:
            raise ValueError(f"attachment path escapes input dir: {rel!r}")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"attachment file not found: {rel!r}")
        target = candidate
        size = os.path.getsize(target)
        if not mime:
            mime = _guess_mime(name)

    return Attachment(name=name, mime_type=mime, path=target, size=size,
                      description=description,
                      extra={k: v for k, v in entry.items()
                             if k not in ("name", "filename", "data", "path",
                                          "mime_type", "mimeType",
                                          "description", "caption")})


def _summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, secret-free view of an attachment spec for diagnostics."""
    return {
        "name": entry.get("name") or entry.get("filename"),
        "mime_type": entry.get("mime_type") or entry.get("mimeType"),
        "has_data": "data" in entry,
        "path": entry.get("path"),
    }


# --------------------------------------------------------------------------- #
# Caller-side helpers (used by the deploy/test harness)
# --------------------------------------------------------------------------- #

def attachment_from_file(path: str, *, name: Optional[str] = None,
                         mime_type: Optional[str] = None,
                         description: str = "") -> Dict[str, Any]:
    """Build an inline-data attachment spec suitable for ``task.json``.

    Used by the host-side signalling harness to ship a local file as part of
    the multimodal prompt: the bytes are base64-encoded into the task payload,
    the node uploads ``task.json`` to ``/app/input``, and the davinci creature
    decodes + writes the bytes back out to its filesystem.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    return {
        "name": name or os.path.basename(path),
        "mime_type": mime_type or _guess_mime(name or path),
        "description": description,
        "data": base64.b64encode(data).decode("ascii"),
    }
