# slack_connector

Real Slack Web API operations via `slack_sdk`. Auth: `SLACK_BOT_TOKEN` (or
`payload.token`).

- Image: `apps-tools:latest`
- Function: `slack_action` (alias `invoke`)

## Operations

`post_message`, `update_message`, `delete_message`, `reply`, `add_reaction`,
`list_channels`, `history`, `upload_file`, `users_list`, `find_channel`,
`auth_test`. Channel names (`#name`) are resolved to IDs automatically.
