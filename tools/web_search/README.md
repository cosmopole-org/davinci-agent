# web_search

Real web search with pluggable backends. Picks the first configured backend, or
force one with `engine`:

| engine       | credential(s)                     |
|--------------|-----------------------------------|
| `tavily`     | `TAVILY_API_KEY`                  |
| `brave`      | `BRAVE_API_KEY`                   |
| `serpapi`    | `SERPAPI_API_KEY`                 |
| `google`     | `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`|
| `duckduckgo` | none (default fallback)           |

- Image: `web-tools:latest`
- Function: `search` (alias `invoke`)

## Payload

`{query, max_results=5, engine?, region?}` → `{ok, engine, query, count, results:[{title,url,snippet}], answer?}`.
