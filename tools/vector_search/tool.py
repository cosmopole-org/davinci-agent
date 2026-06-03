"""vector_search tool creature — real semantic retrieval over a vector index.

Embedding backends (auto-selected; override with ``VECTOR_BACKEND`` or the
``backend`` payload field):

* ``sentence_transformers`` — local model (``VECTOR_MODEL``, default
  ``all-MiniLM-L6-v2``) producing dense semantic embeddings. Used if the
  package is installed.
* ``openai`` — ``OPENAI_API_KEY`` set; uses ``text-embedding-3-small``.
* ``hashing`` — always available fallback: a deterministic
  ``HashingVectorizer`` (scikit-learn) producing L2-normalised dense vectors.
  Real cosine-similarity retrieval with zero external dependencies.

Collections are persisted as JSON under ``VECTOR_DATA_DIR`` (default
``/app/data``) so an index survives across signals.

Signal payload (function ``vector_search`` or ``invoke``)::

    {
      "action": "search" | "index" | "delete" | "reset" | "list",
      "collection": "default",
      "query": "...",                 # for search
      "top_k": 5,
      "documents": ["text", {"id": "x", "text": "...", "metadata": {...}}],
      "ids": ["x"]                     # for delete
    }

A one-shot search is supported by passing both ``query`` and ``documents``
without a persisted collection.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

DATA_DIR = os.environ.get("VECTOR_DATA_DIR", "/app/data")
DEFAULT_DIM = 1024


# --------------------------------------------------------------------------- #
# Embedding backends
# --------------------------------------------------------------------------- #

_ST_MODEL = None


def _select_backend(payload: dict) -> str:
    forced = (payload.get("backend") or os.environ.get("VECTOR_BACKEND") or "").lower().strip()
    if forced:
        return forced
    try:
        import sentence_transformers  # noqa: F401
        return "sentence_transformers"
    except Exception:
        pass
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "hashing"


def _embed(texts, backend: str) -> np.ndarray:
    if backend == "sentence_transformers":
        return _embed_st(texts)
    if backend == "openai":
        return _embed_openai(texts)
    return _embed_hashing(texts)


def _embed_st(texts) -> np.ndarray:
    global _ST_MODEL
    from sentence_transformers import SentenceTransformer
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(os.environ.get("VECTOR_MODEL", "all-MiniLM-L6-v2"))
    vecs = _ST_MODEL.encode(list(texts), normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def _embed_openai(texts) -> np.ndarray:
    import requests
    model = os.environ.get("VECTOR_MODEL", "text-embedding-3-small")
    r = requests.post("https://api.openai.com/v1/embeddings", timeout=30,
                      headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                      json={"model": model, "input": list(texts)})
    r.raise_for_status()
    vecs = np.asarray([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
    return _l2(vecs)


def _embed_hashing(texts) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer
    dim = int(os.environ.get("VECTOR_DIM", DEFAULT_DIM))
    vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2",
                            ngram_range=(1, 2), stop_words="english")
    mat = vec.transform(list(texts))
    return np.asarray(mat.todense(), dtype=np.float32)


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# --------------------------------------------------------------------------- #
# Persistent collection store
# --------------------------------------------------------------------------- #

def _collection_path(name: str) -> str:
    safe = hashlib.sha1(name.encode()).hexdigest()[:16]
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"col_{safe}.json")


def _load(name: str) -> dict:
    path = _collection_path(name)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"name": name, "backend": None, "items": []}


def _save(name: str, store: dict) -> None:
    with open(_collection_path(name), "w", encoding="utf-8") as fh:
        json.dump(store, fh)


def _normalise_docs(documents) -> list:
    docs = []
    for i, d in enumerate(documents or []):
        if isinstance(d, str):
            docs.append({"id": f"doc-{i}", "text": d, "metadata": {}})
        elif isinstance(d, dict):
            docs.append({"id": str(d.get("id", f"doc-{i}")), "text": str(d.get("text", "")),
                         "metadata": d.get("metadata", {})})
    return [d for d in docs if d["text"]]


def _search_vectors(query_vec: np.ndarray, matrix: np.ndarray, top_k: int):
    sims = matrix @ query_vec  # both L2-normalised → cosine similarity
    order = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in order]


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #

def _do_index(name: str, documents, backend: str) -> dict:
    docs = _normalise_docs(documents)
    if not docs:
        return {"ok": False, "error": "no valid documents to index"}
    embs = _embed([d["text"] for d in docs], backend)
    store = _load(name)
    store["backend"] = backend
    by_id = {it["id"]: it for it in store["items"]}
    for d, e in zip(docs, embs):
        d["embedding"] = e.tolist()
        by_id[d["id"]] = d
    store["items"] = list(by_id.values())
    _save(name, store)
    return {"ok": True, "action": "index", "collection": name, "backend": backend,
            "indexed": len(docs), "total": len(store["items"]), "dim": int(embs.shape[1])}


def _do_search(name: str, query: str, top_k: int, backend: str, documents=None) -> dict:
    if documents:  # one-shot search over provided docs
        docs = _normalise_docs(documents)
        embs = _embed([d["text"] for d in docs], backend)
        items = docs
        matrix = _l2(embs)
    else:
        store = _load(name)
        items = store["items"]
        if not items:
            return {"ok": True, "action": "search", "collection": name,
                    "results": [], "note": "collection empty"}
        backend = store.get("backend") or backend
        matrix = _l2(np.asarray([it["embedding"] for it in items], dtype=np.float32))

    qvec = _l2(_embed([query], backend))[0]
    hits = _search_vectors(qvec, matrix, top_k)
    results = [{"id": items[i]["id"], "score": round(score, 6),
                "text": items[i]["text"][:2000], "metadata": items[i].get("metadata", {})}
               for i, score in hits]
    return {"ok": True, "action": "search", "collection": name, "backend": backend,
            "query": query, "count": len(results), "results": results}


def _do_delete(name: str, ids) -> dict:
    store = _load(name)
    before = len(store["items"])
    drop = set(str(i) for i in (ids or []))
    store["items"] = [it for it in store["items"] if it["id"] not in drop]
    _save(name, store)
    return {"ok": True, "action": "delete", "collection": name,
            "deleted": before - len(store["items"]), "total": len(store["items"])}


def _do_reset(name: str) -> dict:
    path = _collection_path(name)
    if os.path.isfile(path):
        os.remove(path)
    return {"ok": True, "action": "reset", "collection": name}


def invoke(function_name: str, payload: dict) -> dict:
    payload = payload or {}
    name = str(payload.get("collection", "default"))
    action = (payload.get("action") or ("search" if (payload.get("query") or payload.get("task")) else "index")).lower()
    backend = _select_backend(payload)

    try:
        if action == "index":
            return _do_index(name, payload.get("documents"), backend)
        if action == "delete":
            return _do_delete(name, payload.get("ids"))
        if action in ("reset", "clear"):
            return _do_reset(name)
        if action == "list":
            store = _load(name)
            return {"ok": True, "action": "list", "collection": name,
                    "total": len(store["items"]), "ids": [it["id"] for it in store["items"]]}
        # default: search
        query = str(payload.get("query") or payload.get("task") or "")
        if not query:
            return {"ok": False, "error": "no 'query' provided for search"}
        top_k = max(1, min(int(payload.get("top_k", 5) or 5), 100))
        return _do_search(name, query, top_k, backend, documents=payload.get("documents"))
    except Exception as exc:
        return {"ok": False, "action": action, "error": str(exc)}


if __name__ == "__main__":
    docs = ["Davinci is an orchestration-first enterprise agent.",
            "Caspar runs creatures across six VM runtimes.",
            "Tool creatures respond to signals over the action protocol."]
    print(json.dumps(invoke("vector_search",
          {"query": "how do creatures communicate", "documents": docs, "top_k": 2}), indent=2))
