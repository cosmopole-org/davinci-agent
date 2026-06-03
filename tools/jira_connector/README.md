# jira_connector

Real Jira Cloud REST API (v3) operations. Auth via `JIRA_URL`, `JIRA_EMAIL`,
`JIRA_API_TOKEN` (or `base_url`/`email`/`api_token` in the payload).

- Image: `apps-tools:latest`
- Function: `jira_action` (alias `invoke`)

## Operations

`get_issue`, `create_issue`, `update_issue`, `add_comment`, `search` (JQL),
`list_transitions`, `transition`, `assign`, `delete_issue`, `myself`.
Plain-text descriptions/comments are wrapped in Atlassian Document Format.
