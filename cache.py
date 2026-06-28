"""
cache.py — semantic LLM response cache, scoped by an exact data fingerprint.

Two-stage matching, not one:
  1. HARD FILTER (exact match): only compare against cache entries from
     the SAME dataframe shape/columns. This makes it structurally
     impossible for two genuinely different datasets to collide, no
     matter how similar their wording happens to be.
  2. SOFT MATCH (semantic): WITHIN that same-dataset group, use embedding
     similarity to catch paraphrased questions about it.

Why both: relying on the embedder alone to notice a single differing
digit in an otherwise near-identical sentence (e.g. "shape (8, 3)" vs
"shape (6, 3)") isn't reliable — small embedding models don't always
weight that difference heavily enough. The fingerprint filter removes
that risk entirely instead of trying to out-tune a threshold.
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer

CACHE_DB_PATH = "cache.db"
SIMILARITY_THRESHOLD = 0.97

_model = None


def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def init_cache_db():
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TEXT NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def _messages_to_text(messages: list) -> str:
    """Embed only the most recent message — see earlier note: embedding
    the full growing history let the long static system prompt dilute
    the comparison as conversations grew."""
    last = messages[-1]
    return f"{last['role']}: {last['content']}"


def _cosine_similarity(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_cached_response(messages: list, fingerprint: str):
    """
    Only compares against cache entries with the SAME fingerprint —
    i.e. the same dataframe shape/columns. Different datasets can never
    match each other here, regardless of how similar their text is.
    """
    text = _messages_to_text(messages)
    query_vec = get_embedder().encode(text)

    conn = sqlite3.connect(CACHE_DB_PATH)
    rows = conn.execute(
        "SELECT id, embedding, response FROM llm_cache WHERE fingerprint = ?",
        (fingerprint,),
    ).fetchall()
    conn.close()

    best_score, best_row = -1.0, None
    for row_id, emb_json, response in rows:
        stored_vec = np.array(json.loads(emb_json))
        score = _cosine_similarity(query_vec, stored_vec)
        if score > best_score:
            best_score, best_row = score, (row_id, response)

    if best_row and best_score >= SIMILARITY_THRESHOLD:
        conn = sqlite3.connect(CACHE_DB_PATH)
        conn.execute("UPDATE llm_cache SET hit_count = hit_count + 1 WHERE id = ?", (best_row[0],))
        conn.commit()
        conn.close()
        return best_row[1], best_score

    return None, (best_score if best_row else None)


def set_cached_response(messages: list, response: str, fingerprint: str):
    text = _messages_to_text(messages)
    vec = get_embedder().encode(text)
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(
        "INSERT INTO llm_cache (fingerprint, text, embedding, response, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (fingerprint, text, json.dumps(vec.tolist()), response, now),
    )
    conn.commit()
    conn.close()


def get_cache_stats():
    conn = sqlite3.connect(CACHE_DB_PATH)
    row = conn.execute(
        "SELECT COUNT(*) as entries, COALESCE(SUM(hit_count), 0) as total_hits FROM llm_cache"
    ).fetchone()
    conn.close()
    return {"cached_entries": row[0], "total_cache_hits": row[1]}