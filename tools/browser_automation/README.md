# browser_automation

Real headless browser automation via Playwright (Chromium/Firefox/WebKit).

- Image: `browser-tools:latest` (built from the Playwright Python base image)
- Function: `automate` (alias `invoke`)

## Payload

`{url?, browser=chromium, headless=true, viewport?, user_agent?, timeout_ms=15000,
ignore_https_errors?, continue_on_error?, steps:[...]}`

Each step has an `action` (`goto`, `click`, `fill`, `type`, `press`, `hover`,
`check`, `select_option`, `scroll`, `wait_for_selector`, `wait_for_timeout`,
`wait_for_load_state`, `screenshot`, `pdf`, `text`, `inner_html`,
`get_attribute`, `content`, `title`, `url`, `evaluate`, `go_back`, `reload`).

Returns `{ok, final:{url,title}, steps:[...]}`; screenshots/pdf are base64.
