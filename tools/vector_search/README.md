# vector_search

Real semantic retrieval over a persistent vector index (cosine similarity).

- Image: `data-tools:latest`
- Function: `vector_search` (alias `invoke`)

## Backends (auto-selected, override via `VECTOR_BACKEND`/`backend`)

| backend                 | requirement                              |
|-------------------------|------------------------------------------|
| `sentence_transformers` | `sentence-transformers` installed        |
| `openai`                | `OPENAI_API_KEY`                         |
| `hashing`               | none — scikit-learn HashingVectorizer    |

## Payload

`{action: search|index|delete|reset|list, collection="default", query?, top_k=5, documents?, ids?}`.
Pass `query`+`documents` together for a one-shot search; collections persist
under `VECTOR_DATA_DIR` (`/app/data`).
