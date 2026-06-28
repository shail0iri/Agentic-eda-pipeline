"""
agent.py — the THINK/ACT/OBSERVE loop, as a reusable function.

This is the exact same logic from layer0_agent.py, just no longer tied to
a terminal: it takes a DataFrame in, returns a list of step records out.
Nothing about the *agent* changed — only how it's called.
"""

import os
import io
import contextlib
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"
MAX_STEPS = 5

SYSTEM_PROMPT = """You are an EDA (exploratory data analysis) agent.
You have access to a pandas DataFrame called `df`, already loaded in memory.

On each turn:
- Decide ONE next analysis step (e.g. check nulls, check dtypes, check a distribution).
- Respond with ONLY a Python code block that does that one step and prints results.
- Do not re-load the CSV. `df` already exists.
- When you believe the analysis is complete, respond with exactly: DONE
  followed by a short plain-English summary of findings.
"""


def run_code(code: str, local_vars: dict) -> str:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {}, local_vars)
    except Exception as e:
        return f"ERROR while executing code:\n{e}"
    return buf.getvalue() or "(code ran, no printed output)"


def run_agent_loop(df: pd.DataFrame) -> dict:
    """
    Runs the full THINK/ACT/OBSERVE loop against the given dataframe.
    Returns a dict with every step taken and the final status —
    this is what the API will hand back as JSON.
    """
    local_vars = {"df": df, "pd": pd}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"The dataframe `df` has columns: {list(df.columns)} "
                                     f"and shape {df.shape}. What's your first analysis step?"}
    ]

    steps = []
    finished = False
    summary = None

    for step_num in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=MODEL, max_tokens=500, messages=messages,
        )
        reply_text = response.choices[0].message.content

        if "```" in reply_text:
            code = reply_text.split("```")[1].replace("python", "", 1).strip()
        else:
            code = reply_text.strip()

        if code.startswith("DONE"):
            finished = True
            summary = code
            break

        result = run_code(code, local_vars)
        steps.append({"step": step_num, "code": code, "result": result})

        messages.append({"role": "assistant", "content": reply_text})
        messages.append({"role": "user", "content": f"Output of your code:\n{result}\n\nWhat's next?"})

    return {
        "steps": steps,
        "finished": finished,
        "summary": summary,
        "stopped_reason": "model said DONE" if finished else "max steps reached",
    }
