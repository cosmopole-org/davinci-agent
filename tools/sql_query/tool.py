"""sql_query tool creature — execute SQL against a real database via SQLAlchemy.

Connects with a SQLAlchemy URL and runs one or more SQL statements, returning
columns + typed rows for result sets and ``rowcount`` for DML/DDL. Drivers for
SQLite (built in), PostgreSQL (``psycopg2``) and MySQL (``PyMySQL``) ship in the
image; any other SQLAlchemy-supported driver works if installed.

Connection resolution order:
    1. ``payload['database_url']``
    2. ``DATABASE_URL`` env var
    3. ``sqlite:////app/data/davinci.sqlite`` (persistent local default)

Signal payload (function ``query`` or ``invoke``)::

    {
      "sql": "SELECT * FROM t WHERE id = :id",   # required (string or list)
      "params": {"id": 1},                        # bound parameters
      "database_url": "postgresql+psycopg2://...",# optional
      "read_only": true,                          # reject writes if true
      "max_rows": 1000,
      "transaction": true                         # wrap multi-statement in a txn
    }
"""

from __future__ import annotations

import json
import os
import re

DEFAULT_URL = "sqlite:////app/data/davinci.sqlite"
DEFAULT_MAX_ROWS = 1000

_WRITE_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE|GRANT|REVOKE|ATTACH|PRAGMA)\b",
    re.IGNORECASE,
)


def _resolve_url(payload: dict) -> str:
    return payload.get("database_url") or os.environ.get("DATABASE_URL") or DEFAULT_URL


def _split_statements(sql) -> list:
    if isinstance(sql, (list, tuple)):
        return [str(s).strip() for s in sql if str(s).strip()]
    # naive split on ';' that respects single-quoted strings
    parts, buf, in_str = [], [], False
    for ch in str(sql):
        if ch == "'":
            in_str = not in_str
        if ch == ";" and not in_str:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _jsonable(value):
    import datetime
    import decimal
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return bytes(value).decode("utf-8")
        except Exception:
            import base64
            return {"__bytes_b64__": base64.b64encode(bytes(value)).decode()}
    return value


def invoke(function_name: str, payload: dict) -> dict:
    payload = payload or {}
    sql = payload.get("sql") or payload.get("query") or payload.get("task")
    if not sql:
        return {"ok": False, "error": "no 'sql' provided"}

    statements = _split_statements(sql)
    read_only = bool(payload.get("read_only", False))
    if read_only:
        bad = [s for s in statements if _WRITE_RE.match(s)]
        if bad:
            return {"ok": False, "error": "read_only mode: write/DDL statement rejected",
                    "rejected": bad[:3]}

    max_rows = max(1, min(int(payload.get("max_rows", DEFAULT_MAX_ROWS) or DEFAULT_MAX_ROWS), 100_000))
    params = payload.get("params") or {}
    url = _resolve_url(payload)

    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.engine import Result
    except Exception as exc:
        return {"ok": False, "error": f"sqlalchemy unavailable: {exc}"}

    if url.startswith("sqlite:///") and "/app/data/" in url:
        os.makedirs("/app/data", exist_ok=True)

    try:
        engine = create_engine(url, future=True, pool_pre_ping=True)
    except Exception as exc:
        return {"ok": False, "error": f"could not create engine for url: {exc}"}

    results = []
    use_txn = bool(payload.get("transaction", len(statements) > 1))
    try:
        with engine.connect() as conn:
            tx = conn.begin() if use_txn else None
            try:
                for stmt in statements:
                    res: "Result" = conn.execute(text(stmt), params)
                    entry = {"statement": stmt[:500]}
                    if res.returns_rows:
                        cols = list(res.keys())
                        rows = []
                        for i, row in enumerate(res):
                            if i >= max_rows:
                                entry["truncated"] = True
                                break
                            rows.append({c: _jsonable(v) for c, v in zip(cols, row)})
                        entry.update({"columns": cols, "rows": rows, "rowcount": len(rows)})
                    else:
                        entry["rowcount"] = res.rowcount
                        if res.lastrowid is not None:
                            entry["lastrowid"] = res.lastrowid
                    results.append(entry)
                if tx is not None:
                    tx.commit()
            except Exception:
                if tx is not None:
                    tx.rollback()
                raise
    except Exception as exc:
        return {"ok": False, "url": _redact(url), "error": str(exc),
                "statements_attempted": len(results) + 1}
    finally:
        engine.dispose()

    return {"ok": True, "function": function_name, "url": _redact(url),
            "dialect": url.split(":", 1)[0], "statement_count": len(statements),
            "results": results}


def _redact(url: str) -> str:
    return re.sub(r"//([^:/@]+):([^@]+)@", r"//\1:***@", url)


if __name__ == "__main__":
    demo = ["CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, name TEXT)",
            "INSERT INTO t (name) VALUES ('davinci'), ('caspar')",
            "SELECT * FROM t ORDER BY id"]
    print(json.dumps(invoke("query", {"sql": demo, "database_url": "sqlite:///:memory:"}), indent=2))
