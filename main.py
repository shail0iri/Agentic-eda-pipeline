"""
main.py — FastAPI entrypoint.

Run with:  uvicorn main:app --reload
Then open: http://127.0.0.1:8000/docs

Two endpoints now:
  POST /analyze  -> upload a CSV, starts a NEW session, returns session_id + findings
  POST /ask      -> continue an EXISTING session with a follow-up question
"""

import io
import os
import uuid
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from graph import run_agent_graph
import db
import cache
import evaluation

app = FastAPI(title="Agentic EDA API")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


@app.on_event("startup")
def on_startup():
    db.init_db()
    cache.init_cache_db()


@app.get("/cache-stats")
def cache_stats():
    """See how much the cache is actually saving you — useful while testing."""
    return cache.get_cache_stats()


@app.get("/eval")
def run_eval():
    """
    Runs the automated test suite against the LIVE agent.
    NOTE: this makes real Groq API calls (one full agent run per test
    case) — it's a periodic sanity check, not something to hit constantly.
    """
    return evaluation.run_eval_suite()


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """Upload a CSV. Creates a new session and runs the first analysis pass."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't parse CSV: {e}")

    session_id = str(uuid.uuid4())
    csv_path = os.path.join(DATA_DIR, f"{session_id}.csv")
    with open(csv_path, "wb") as f:
        f.write(raw_bytes)

    result = run_agent_graph(df)  # messages=None -> fresh conversation

    db.create_session(
        session_id=session_id,
        csv_path=csv_path,
        messages=result["messages"],
        steps=result["steps"],
        finished=result["finished"],
        summary=result["summary"],
    )

    return {
        "session_id": session_id,
        "steps": result["steps"],
        "finished": result["finished"],
        "summary": result["summary"],
        "stopped_reason": result["stopped_reason"],
    }


class AskRequest(BaseModel):
    session_id: str
    question: str


@app.post("/ask")
def ask_followup(request: AskRequest):
    """Continue an existing session with a follow-up question about the same dataset."""
    session = db.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    df = pd.read_csv(session["csv_path"])

    messages = session["messages"] + [
        {"role": "user", "content": f"Follow-up question from the user: {request.question}"}
    ]

    result = run_agent_graph(df, messages=messages)

    all_steps = session["steps"] + result["steps"]
    db.update_session(
        session_id=request.session_id,
        messages=result["messages"],
        steps=all_steps,
        finished=result["finished"],
        summary=result["summary"],
    )

    return {
        "session_id": request.session_id,
        "new_steps": result["steps"],
        "finished": result["finished"],
        "summary": result["summary"],
        "stopped_reason": result["stopped_reason"],
    }