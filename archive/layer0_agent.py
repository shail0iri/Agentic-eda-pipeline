"""
LAYER 0 — Bare-metal agentic EDA loop.
No LangGraph. No sandboxing. No DB. Just the raw pattern:

    THINK  -> ask Claude what to do next
    ACT    -> run the Python code Claude wrote
    OBSERVE -> capture what happened
    REPEAT -> feed the observation back in

This is intentionally unsafe (exec() on raw model output) and intentionally
simple. The whole point is to see the loop with nothing hidden, before we
let a framework hide it for us.

Run with:  python layer0_agent.py your_data.csv
"""

import os
import sys
import io
import contextlib
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads .env file in this folder into environment variables

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"  # swap for any model your Groq account has access to
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
    """ACT + OBSERVE: execute the model's code, capture stdout, return it as text."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {}, local_vars)
    except Exception as e:
        return f"ERROR while executing code:\n{e}"
    return buf.getvalue() or "(code ran, no printed output)"


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    df = pd.read_csv(csv_path)
    local_vars = {"df": df, "pd": pd}

    # conversation history we feed back to the model each turn.
    # Groq's API is OpenAI-style: system prompt goes IN the messages list,
    # not as a separate parameter (unlike Anthropic's API).
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"The dataframe `df` is loaded from {csv_path}. "
                                     f"It has columns: {list(df.columns)} and shape {df.shape}. "
                                     f"What's your first analysis step?"}
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n=== STEP {step}: THINK ===")
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=500,
            messages=messages,
        )
        reply_text = response.choices[0].message.content
        print(reply_text)

        # extract code from a ```python ... ``` block (basic, no framework magic)
        if "```" in reply_text:
            code = reply_text.split("```")[1]
            code = code.replace("python", "", 1).strip()
        else:
            code = reply_text.strip()

        # check DONE *after* unwrapping the fence — the model sometimes
        # puts "DONE" inside a code block even though the prompt says not to
        if code.startswith("DONE"):
            print("\n=== AGENT FINISHED ===")
            print(code)
            break

        print(f"\n=== STEP {step}: ACT ===")
        result = run_code(code, local_vars)
        print(f"\n=== STEP {step}: OBSERVE ===")
        print(result)

        # feed model's own reply + the execution result back into history
        messages.append({"role": "assistant", "content": reply_text})
        messages.append({"role": "user", "content": f"Output of your code:\n{result}\n\nWhat's next?"})
    else:
        print("\n=== MAX STEPS REACHED ===")


if __name__ == "__main__":
    main()