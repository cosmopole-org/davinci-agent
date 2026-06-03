# fetch_url

Real HTTP fetch with readable content extraction.

- Image: `web-tools:latest`
- Function: `fetch` (alias `invoke`)

## Payload

`{url, method=GET, headers?, params?, data?, json?, timeout=20, max_bytes=2MB, extract=true}`

Returns `{ok, status, content_type, headers, bytes, ...}` plus one of:
`json` (parsed), `content:{title,description,text,links}` (HTML), `text`, or
`binary`/`base64`.
