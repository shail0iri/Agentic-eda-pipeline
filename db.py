"""
db.py — SQLite session storage.

Each "session" = one uploaded CSV + its entire conversation history with
the agent (messages) + every step taken so far. This is what lets a user
come back and ask a follow-up question without re-uploading or losing
what the agent already found.

We store messages/steps as JSON text in a column — SQLite has no native
list/dict type, so this is the simplest correct way to persist them.
"""

import sqlite3
import json
from datetime import datetime, timezone

DB_PATH = "sessions.db"


def init_db():
    """Create the sessions table if it doesn't exist yet. Call once at startup."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            csv_path TEXT NOT NULL,
            messages TEXT NOT NULL,
            steps TEXT NOT NULL,
            finished INTEGER NOT NULL,
            summary TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def create_session(session_id: str, csv_path: str, messages: list, steps: list,
                    finished: bool, summary):
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (session_id, csv_path, messages, steps, finished, summary, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, csv_path, json.dumps(messages), json.dumps(steps),
         int(finished), summary, now, now),
    )
    conn.commit()
    conn.close()


def update_session(session_id: str, messages: list, steps: list, finished: bool, summary):
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE sessions SET messages = ?, steps = ?, finished = ?, summary = ?, updated_at = ? "
        "WHERE session_id = ?",
        (json.dumps(messages), json.dumps(steps), int(finished), summary, now, session_id),
    )
    conn.commit()
    conn.close()


def get_session(session_id: str):
    """Returns a dict with the session's data, or None if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "session_id": row["session_id"],
        "csv_path": row["csv_path"],
        "messages": json.loads(row["messages"]),
        "steps": json.loads(row["steps"]),
        "finished": bool(row["finished"]),
        "summary": row["summary"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
