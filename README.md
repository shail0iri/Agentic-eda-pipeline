# Agentic EDA Pipeline

A production-grade API that analyzes CSV files using an LLM agent. Upload a dataset, and instead of you writing pandas code manually, the agent decides what to check, writes and runs the code itself, looks at the results, and decides what to check next — looping until the analysis is complete.

Built layer by layer to understand every piece, not just to ship something that works.

**GitHub:** https://github.com/shail0iri/Agentic-eda-pipeline

---

## What it does

- Upload any CSV via `POST /analyze` — the agent runs a full exploratory analysis and returns step-by-step findings including generated charts (base64 PNG)
- Continue the same analysis with follow-up questions via `POST /ask` — the agent remembers everything it found previously
- Semantic caching means repeated or paraphrased questions skip the LLM call entirely
- Run `GET /eval` to verify the agent actually works against synthetic test cases with planted issues

---

## Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI (main.py)               │
│  POST /analyze  POST /ask  GET /eval    │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│      LangGraph State Machine            │
│                                         │
│   ┌──────────┐     ┌──────────────┐     │
│   │think_node│────▶│  act_node    │     │
│   │ (Groq)   │◀────│  (sandbox)   │     │
│   └────┬─────┘     └──────────────┘     │
│        │ DONE                           │
│        ▼                                │
│       END                               │
└─────────────────────────────────────────┘
         │                    │
┌────────▼────────┐  ┌────────▼────────┐
│  Semantic Cache │  │ Session Memory  │
│  (SQLite +      │  │ (SQLite db.py)  │
│  sentence-      │  └─────────────────┘
│  transformers)  │
└─────────────────┘
```

### The agent loop (ReAct/CodeAct pattern)

```
THINK  → model decides the single most useful next step
ACT    → exec() that code inside the sandbox
OBSERVE → capture stdout + matplotlib figures as base64 PNGs
         → feed result back into the conversation
REPEAT → until model says DONE or MAX_STEPS (8) is hit
```

The key design choice: reflection lives in the **system prompt**, not in separate graph nodes. "Look at what you've found SO FAR... let your findings drive your next move" achieves reasoning-driven behavior without the complexity of a Planner→Analyzer→Executor→Critic→Reporter chain. This is a deliberate simplification, not an oversight.

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/analyze` | Upload CSV, start new session, run agent |
| `POST` | `/ask` | Continue existing session with follow-up question |
| `GET` | `/eval` | Run automated test suite (costs ~24 Groq API calls) |
| `GET` | `/cache-stats` | View cache hit counts |

### Example: analyze a CSV

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@your_data.csv"
```

Response includes:
```json
{
  "session_id": "e040e413-...",
  "steps": [
    {
      "step": 1,
      "code": "print(df.describe())",
      "result": "...",
      "images": ["iVBORw0KGgo..."],
      "from_cache": false,
      "cache_similarity": 0.61
    }
  ],
  "finished": true,
  "summary": "DONE\nFound 2 nulls in age column..."
}
```

### Example: ask a follow-up

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id": "e040e413-...", "question": "How should I handle that salary outlier?"}'
```

---

## Tech stack

| Tool | Purpose | Why |
|------|---------|-----|
| Groq (llama-3.3-70b-versatile) | LLM inference | Fast, generous free tier |
| FastAPI | Web framework | Auto-generates `/docs` test UI |
| LangGraph | Agent loop structure | Formalizes THINK/ACT state machine |
| SQLite | Session persistence | Zero setup, right for this scale |
| sentence-transformers (all-MiniLM-L6-v2) | Local embeddings for cache | Free, offline, no per-call cost |
| matplotlib (Agg backend) | Chart capture | Agg = headless server rendering |
| Docker | Containerization | CPU-only torch = 192MB vs 2.5GB+ |

---

## Semantic caching — how it works

Two-stage matching, not one:

1. **Hard filter (exact match):** only compare against cache entries with the same `(columns, shape, PROMPT_VERSION)` fingerprint. Different datasets can never collide here, regardless of how similar their wording is.

2. **Soft match (semantic):** within the same-dataset group, use cosine similarity of the last message's embedding at threshold 0.97 to catch paraphrased questions.

`PROMPT_VERSION` is a short MD5 hash of the system prompt text — any edit to the prompt automatically invalidates all old cache entries without needing to manually clear the DB.

Why local embeddings instead of ChromaDB: functionally equivalent for this use case, zero cost, works offline.

---

## Code sandbox — what it blocks

Two layers before anything runs with `exec()`:

**Layer 1 — Static pattern check** (before execution):
blocks `os.`, `open(`, `eval(`, `exec(`, `subprocess`, `shutil`, `__import__`, `socket`, `pickle`, `ctypes`, `requests`, `urllib` and more.

**Layer 2 — Restricted builtins**:
`exec()` runs with a curated safe subset of builtins. Custom `__import__` only allows: `pandas`, `numpy`, `matplotlib`, `math`, `statistics`, `json`, `re`, `datetime`, `itertools`, `collections`.

**Verified with test_sandbox.py:**
```
=== Dangerous code (should be BLOCKED) ===
[BLOCKED] import os
[BLOCKED] open('file.txt').read()
[BLOCKED] eval('1+1')
[BLOCKED] __import__('subprocess').run(['ls'])
[BLOCKED] import shutil

=== Safe code (should run normally) ===
print(df.describe())  → ✅ works
import numpy as np    → ✅ works
```

**Honest limitation:** not bulletproof against adversarial dunder-attribute chaining. Protects against realistic LLM mistakes, not a determined attacker. True hardened isolation means Docker-level process isolation per execution.

---

## Eval suite — `GET /eval`

Three synthetic test cases with deliberately planted issues:

| Case | Planted issue | What we check for |
|------|--------------|-------------------|
| `missing_values` | 2 nulls in `age` column | keywords: null, missing, isnull, nan + column name |
| `outlier` | $9,999,999 salary in otherwise normal range | keywords: outlier, extreme, anomal, unusual + column name |
| `duplicate_rows` | 2 exact duplicate rows | keywords: duplicate, drop_duplicates |

**Final score: 3/3**

The eval suite found two real bugs during development:
- **Cross-dataset cache collision** — all three test cases used the same column names, so the semantic cache served the `outlier` case's answer to the `duplicate_rows` case. The fix (hard fingerprint filter) came directly from this failure.
- **Outlier keyword mismatch** — the model printed `9.999999e+06` (scientific notation) but the eval checked for the literal string `9999999`. Fixed by checking for semantic words (outlier, extreme) instead of exact number strings.

---

## Bugs fixed — the ones worth reading

These came from real failures, not hypotheticals.

**DONE detection** — the model wrapped `DONE` inside a code fence (` ```python\nDONE\n...` ```) even though the system prompt said not to. A `startswith("DONE")` check on the raw reply missed it entirely — the fix was scanning every line of the reply rather than assuming a fixed position. Lesson: never trust an LLM to follow an exact output format reliably.

**matplotlib in a headless server** — `plt.show()` does nothing when there's no display. Charts were silently discarded. Fix: `matplotlib.use("Agg")` + manually capture every figure to a BytesIO buffer, encode as base64, return in the JSON response.

**Semantic cache getting stuck** — embedding the entire growing conversation let the long, static system prompt dominate the vector. Steps 2–8 all matched step 1's cache entry because the shared prefix outweighed the genuinely new content. Fix: embed only the last message, not the full history.

**exec() namespace and nested functions** — the model occasionally wrote nested helper functions referencing `pd`, which raised `NameError` because `pd` was in the locals dict but Python resolves free variables in nested functions via globals only. Fix: merge sandbox globals and local_vars into one shared namespace.

**PROMPT_VERSION cache invalidation** — after editing the system prompt to fix the duplicate-row blind spot, the old cached answers were still being served. Fix: include a short MD5 hash of the system prompt in the cache fingerprint — any edit automatically busts all cached entries.

**GPU torch filling EC2 disk** — `sentence-transformers` pulls full GPU torch (2.5GB+) by default. A t3.micro has 8GB total disk. Fix: a separate Dockerfile `RUN` step installs CPU-only torch first (`--index-url https://download.pytorch.org/whl/cpu`) before `requirements.txt`, keeping the image at ~1GB instead of 3.5GB+.

---

## Running locally

**Prerequisites:** Docker Desktop running, a Groq API key.

```powershell
git clone https://github.com/shail0iri/Agentic-eda-pipeline
cd Agentic-eda-pipeline

# Create .env with your key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# Windows: pre-create SQLite files as actual files (not directories)
New-Item -ItemType File -Name sessions.db -ErrorAction SilentlyContinue
New-Item -ItemType File -Name cache.db -ErrorAction SilentlyContinue

docker compose up --build
```

Then open: `http://127.0.0.1:8000/docs`

**Windows gotcha:** if you ever delete `sessions.db` or `cache.db`, always re-create them as empty files with `New-Item` before running `docker compose up`. If Docker starts with those paths missing, it creates directories instead of files, and SQLite will crash on startup.

---

## AWS deployment

1. Launch `t3.micro`, Ubuntu 24.04 LTS
2. Open ports 22, 8000 in Security Group inbound rules
3. Connect via EC2 Instance Connect
4. Install Docker:
   ```bash
   sudo apt-get update && sudo apt-get install -y docker.io docker-compose
   sudo usermod -aG docker ubuntu && newgrp docker
   ```
5. Clone repo, create `.env`, pre-create DB files:
   ```bash
   git clone https://github.com/shail0iri/Agentic-eda-pipeline
   cd Agentic-eda-pipeline
   echo "GROQ_API_KEY=gsk_your_key" > .env
   touch sessions.db cache.db
   ```
6. Run:
   ```bash
   docker compose up -d --build
   ```

For a stable address: allocate an Elastic IP in the EC2 console and associate it with the instance (free while attached to a running instance).

---

## Known limitations

**MAX_STEPS behavior** — Llama 3.3 70B consistently finds "one more thing to check" rather than self-terminating. The hard cap of 8 steps prevents runaway API usage, but the agent often stops at the cap rather than emitting `DONE` naturally. This is a model tendency, not a harness bug.

**Sandbox not bulletproof** — creative payloads chaining `__class__.__bases__` or similar dunder-attribute paths could still reach restricted areas. The sandbox protects against the realistic failure mode (LLM accidentally writing file/network operations) but not a deliberately adversarial input.

**SQLite, not Postgres** — correct for this scale, easy to swap. The original plan explicitly said "SQLite dev → Postgres prod."

**No authentication** — any request can hit `/eval` (which costs ~24 Groq API calls) or flood `/analyze`. A real deployment would add API key auth middleware.

---

## What's next

- Swap SQLite → Postgres for proper concurrent access
- Add API key authentication middleware  
- Add Langfuse observability (see what the agent is spending quota on)
- Docker-level process isolation per code execution (true hardened sandbox)
- A simple frontend (React or plain HTML) to make the demo usable without `/docs`
