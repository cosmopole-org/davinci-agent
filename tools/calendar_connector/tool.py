"""calendar_connector tool creature — real calendar operations.

Three providers, auto-selected (override with ``provider`` / ``CALENDAR_PROVIDER``):

* ``google``  — Google Calendar API. Service-account JSON in
  ``GOOGLE_CALENDAR_SA_JSON`` (raw JSON) or ``GOOGLE_APPLICATION_CREDENTIALS``
  (path), or an OAuth bearer token in ``GOOGLE_OAUTH_TOKEN``.
* ``caldav``  — any CalDAV server (Nextcloud, iCloud, Fastmail …) via
  ``CALDAV_URL`` + ``CALDAV_USERNAME`` + ``CALDAV_PASSWORD``.
* ``ics``     — no credentials; build a valid RFC-5545 iCalendar (.ics) for an
  event so the result is always useful offline.

Signal payload (function ``calendar_action`` or ``invoke``)::

    {"operation": "create_event", "calendar_id": "primary",
     "summary": "Sync", "start": "2026-06-10T15:00:00",
     "end": "2026-06-10T15:30:00", "timezone": "UTC", "attendees": [...]}

Operations: ``list_calendars``, ``list_events``, ``get_event``,
``create_event``, ``update_event``, ``delete_event``, ``generate_ics``.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #

def _select_provider(payload: dict) -> str:
    forced = (payload.get("provider") or os.environ.get("CALENDAR_PROVIDER") or "").lower().strip()
    if forced:
        return forced
    if (os.environ.get("GOOGLE_CALENDAR_SA_JSON") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or os.environ.get("GOOGLE_OAUTH_TOKEN")):
        return "google"
    if os.environ.get("CALDAV_URL"):
        return "caldav"
    return "ics"


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    s = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(s)


# --------------------------------------------------------------------------- #
# Google Calendar
# --------------------------------------------------------------------------- #

def _google_service():
    from googleapiclient.discovery import build
    sa_json = os.environ.get("GOOGLE_CALENDAR_SA_JSON")
    sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    scopes = ["https://www.googleapis.com/auth/calendar"]
    if sa_json or sa_path:
        from google.oauth2 import service_account
        info = json.loads(sa_json) if sa_json else None
        creds = (service_account.Credentials.from_service_account_info(info, scopes=scopes)
                 if info else service_account.Credentials.from_service_account_file(sa_path, scopes=scopes))
        subject = os.environ.get("GOOGLE_CALENDAR_SUBJECT")
        if subject:
            creds = creds.with_subject(subject)
    else:
        from google.oauth2.credentials import Credentials
        creds = Credentials(token=os.environ["GOOGLE_OAUTH_TOKEN"], scopes=scopes)
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _google(op: str, p: dict) -> dict:
    svc = _google_service()
    cal = p.get("calendar_id", "primary")
    if op == "list_calendars":
        items = svc.calendarList().list().execute().get("items", [])
        return {"calendars": [{"id": c["id"], "summary": c.get("summary")} for c in items]}
    if op == "list_events":
        ev = svc.events().list(calendarId=cal, maxResults=int(p.get("max_results", 25)),
                               singleEvents=True, orderBy="startTime",
                               timeMin=p.get("time_min")).execute()
        return {"events": [{"id": e["id"], "summary": e.get("summary"),
                            "start": e.get("start"), "end": e.get("end")}
                           for e in ev.get("items", [])]}
    if op == "get_event":
        return svc.events().get(calendarId=cal, eventId=p["event_id"]).execute()
    if op in ("create_event", "update_event"):
        body = _google_event_body(p)
        if op == "create_event":
            e = svc.events().insert(calendarId=cal, body=body,
                                    sendUpdates=p.get("send_updates", "none")).execute()
        else:
            e = svc.events().patch(calendarId=cal, eventId=p["event_id"], body=body,
                                   sendUpdates=p.get("send_updates", "none")).execute()
        return {"id": e["id"], "htmlLink": e.get("htmlLink"), "status": e.get("status")}
    if op == "delete_event":
        svc.events().delete(calendarId=cal, eventId=p["event_id"]).execute()
        return {"deleted": p["event_id"]}
    raise ValueError(f"unknown google operation '{op}'")


def _google_event_body(p: dict) -> dict:
    tz = p.get("timezone", "UTC")
    body = {}
    if p.get("summary"):
        body["summary"] = p["summary"]
    if p.get("description"):
        body["description"] = p["description"]
    if p.get("location"):
        body["location"] = p["location"]
    if p.get("start"):
        body["start"] = {"dateTime": _parse_dt(p["start"]).isoformat(), "timeZone": tz}
    if p.get("end"):
        body["end"] = {"dateTime": _parse_dt(p["end"]).isoformat(), "timeZone": tz}
    if p.get("attendees"):
        body["attendees"] = [{"email": a} if isinstance(a, str) else a for a in p["attendees"]]
    return body


# --------------------------------------------------------------------------- #
# CalDAV
# --------------------------------------------------------------------------- #

def _caldav_client():
    import caldav
    return caldav.DAVClient(url=os.environ["CALDAV_URL"],
                            username=os.environ.get("CALDAV_USERNAME"),
                            password=os.environ.get("CALDAV_PASSWORD"))


def _caldav_calendar(client, p: dict):
    principal = client.principal()
    calendars = principal.calendars()
    target = p.get("calendar_id")
    if target:
        for c in calendars:
            if c.name == target or str(c.url).rstrip("/").endswith(target):
                return c
    if not calendars:
        raise RuntimeError("no CalDAV calendars available")
    return calendars[0]


def _caldav(op: str, p: dict) -> dict:
    client = _caldav_client()
    if op == "list_calendars":
        cals = client.principal().calendars()
        return {"calendars": [{"name": c.name, "url": str(c.url)} for c in cals]}
    cal = _caldav_calendar(client, p)
    if op == "list_events":
        events = cal.events()
        return {"calendar": cal.name, "count": len(events),
                "events": [{"url": str(e.url), "data": e.data[:1000]} for e in events[:int(p.get("max_results", 25))]]}
    if op in ("create_event", "generate_event"):
        ics = _build_ics(p)
        ev = cal.save_event(ics)
        return {"created": True, "url": str(ev.url), "ics": ics}
    if op == "delete_event":
        ev = cal.event_by_url(p["event_url"]) if p.get("event_url") else None
        if ev is None:
            raise RuntimeError("delete_event requires 'event_url'")
        ev.delete()
        return {"deleted": p["event_url"]}
    raise ValueError(f"unknown caldav operation '{op}'")


# --------------------------------------------------------------------------- #
# ICS generation (always available)
# --------------------------------------------------------------------------- #

def _build_ics(p: dict) -> str:
    from icalendar import Calendar, Event
    cal = Calendar()
    cal.add("prodid", "-//Davinci Agent//calendar_connector//EN")
    cal.add("version", "2.0")
    ev = Event()
    ev.add("uid", p.get("uid", f"{uuid.uuid4()}@davinci"))
    ev.add("summary", p.get("summary", "Untitled event"))
    if p.get("description"):
        ev.add("description", p["description"])
    if p.get("location"):
        ev.add("location", p["location"])
    start = _parse_dt(p["start"]) if p.get("start") else datetime.utcnow()
    end = _parse_dt(p["end"]) if p.get("end") else start + timedelta(
        minutes=int(p.get("duration_minutes", 30)))
    ev.add("dtstart", start)
    ev.add("dtend", end)
    ev.add("dtstamp", datetime.utcnow())
    for a in p.get("attendees", []) or []:
        ev.add("attendee", f"MAILTO:{a}" if isinstance(a, str) else a)
    cal.add_component(ev)
    return cal.to_ical().decode("utf-8")


def _ics(op: str, p: dict) -> dict:
    if op in ("generate_ics", "create_event", "generate_event"):
        return {"ics": _build_ics(p), "note": "no calendar backend configured; generated ICS"}
    raise ValueError(f"operation '{op}' requires a google/caldav backend (none configured)")


def invoke(function_name: str, payload: dict) -> dict:
    payload = payload or {}
    op = payload.get("operation") or payload.get("action") or "create_event"
    provider = _select_provider(payload)
    try:
        if provider == "google":
            data = _google(op, payload)
        elif provider == "caldav":
            data = _caldav(op, payload)
        else:
            data = _ics(op, payload)
        return {"ok": True, "provider": provider, "operation": op,
                "function": function_name, "result": data}
    except Exception as exc:
        return {"ok": False, "provider": provider, "operation": op, "error": str(exc)}


if __name__ == "__main__":
    print(json.dumps(invoke("calendar_action",
          {"operation": "generate_ics", "summary": "Davinci sync",
           "start": "2026-06-10T15:00:00", "end": "2026-06-10T15:30:00",
           "attendees": ["team@example.com"]}), indent=2))
