# sql_query

Execute SQL against a real database via SQLAlchemy. SQLite (built in),
PostgreSQL (`psycopg2`) and MySQL (`PyMySQL`) drivers ship in the image.

- Image: `data-tools:latest`
- Function: `query` (alias `invoke`)

## Connection

`payload.database_url` → `DATABASE_URL` env → `sqlite:////app/data/davinci.sqlite`.

## Payload

`{sql (string|list), params?, database_url?, read_only=false, max_rows=1000, transaction?}`

Returns `{ok, dialect, results:[{statement, columns?, rows?, rowcount, lastrowid?}]}`.
Credentials in the URL are redacted in the response.
