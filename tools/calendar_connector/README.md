# calendar_connector

Real calendar operations across three providers (auto-selected, override with
`provider`/`CALENDAR_PROVIDER`):

| provider | credentials                                                        |
|----------|--------------------------------------------------------------------|
| `google` | `GOOGLE_CALENDAR_SA_JSON` / `GOOGLE_APPLICATION_CREDENTIALS` / `GOOGLE_OAUTH_TOKEN` |
| `caldav` | `CALDAV_URL` + `CALDAV_USERNAME` + `CALDAV_PASSWORD`               |
| `ics`    | none — generates a valid RFC-5545 `.ics` (default fallback)        |

- Image: `apps-tools:latest`
- Function: `calendar_action` (alias `invoke`)

## Operations

`list_calendars`, `list_events`, `get_event`, `create_event`, `update_event`,
`delete_event`, `generate_ics`.
