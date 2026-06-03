# git_tool

Unified git container with multi-function routing over real `git` + the GitHub
REST API.

- Image: `git-tools:latest`
- Functions: `status_commit`, `open_pr` (alias `invoke` → `status_commit`)

## `status_commit`

`{repo_path, repo_url?, action=auto, message?, files?, branch?, author_name?, author_email?, remote=origin, push?}`
→ `{ok, branch, head, status, steps}`. With `message` it stages (`files` or all)
and commits. `action` may be `status|add|commit|diff|log|branch|checkout|init|push|pull`.

## `open_pr`

`{repo_path, repo="owner/name"?, title, body, head?, base=main, push=true, github_token?}`
→ pushes the branch and creates (or locates) a GitHub PR, returning `{ok, number, url}`.

Auth uses `GITHUB_TOKEN`/`GH_TOKEN` (or `payload.github_token`) for clone/push/PR.
