"""
graph.py — Layer 2: same brain, now built as a LangGraph state machine,
with a prompt that asks the model to REASON about findings instead of
running a fixed checklist.

Two nodes:
  think_node -> decide next step (or DONE), based on everything found so far
  act_node   -> execute that step, record the result

Conditional edge after think_node: if the model said DONE, go to END.
Otherwise go to act_node, which always loops back to think_node.
This loop *is* what LangGraph formalizes — same shape as your for-loop,
just expressed as a graph so it can later support branching, retries,
and persisted state across requests.
"""

import os
import io
import base64
import contextlib
import warnings
import hashlib
from typing import TypedDict, List, Optional
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END

import matplotlib
matplotlib.use("Agg")  # no GUI in a server — render to memory instead of a window
import matplotlib.pyplot as plt

import cache
import sandbox

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"
MAX_STEPS = 8

SYSTEM_PROMPT = """You are an EDA (exploratory data analysis) agent.
You have access to a pandas DataFrame called `df`, already loaded in memory.

On each turn, look at what you've found SO FAR (not just the last result —
everything observed in this conversation) and decide the single most useful
next step. Do NOT follow a fixed checklist. Let your findings drive your
next move. For example:
- If you find missing values, consider investigating which rows/why, or
  whether to flag a strategy to handle them.
- If a numeric column looks skewed, consider checking its distribution
  more closely or flagging a transform.
- If a categorical column has many rare values, consider grouping them.
- Consider checking for exact duplicate rows (df.duplicated().sum()) —
  this is easy to overlook since it doesn't show up in describe() or info().
- If you notice a value that looks unusually extreme compared to the rest
  of the data (e.g. in describe() output, a max/mean far outside the
  typical range), EXPLICITLY say so in your comments — call it out as a
  likely outlier or anomaly by name. Don't just observe an odd number
  silently and move on to the next check.

Respond with ONLY a Python code block that performs your chosen step and
prints results. Do not re-load the CSV; `df` already exists.

If a chart would genuinely help (e.g. a distribution or scatter plot), you
may create one with matplotlib. It will be automatically captured for the
user — you do not need to call plt.show() or save it yourself.

IMPORTANT pandas note: avoid chained assignment with inplace=True, e.g.
df['col'].fillna(value, inplace=True) — this can SILENTLY fail to modify
the dataframe under pandas' Copy-on-Write behavior. Use one of these
instead: df['col'] = df['col'].fillna(value)  OR  df.fillna({'col': value}, inplace=True)

When you genuinely have nothing more useful to check, respond with exactly:
DONE
followed by a short plain-English summary of everything you found and why
you stopped there.
"""

# A short fingerprint of the CURRENT system prompt text. Any edit to
# SYSTEM_PROMPT changes this hash, which we fold into the cache fingerprint
# below — so old cached answers from a previous prompt version can never
# get served under a new one, without needing to manually clear cache.db.
PROMPT_VERSION = hashlib.md5(SYSTEM_PROMPT.encode()).hexdigest()[:8]


# --- State schema: what flows through the graph ---
class AgentState(TypedDict):
    df: pd.DataFrame
    messages: List[dict]
    steps: List[dict]
    finished: bool
    summary: Optional[str]
    step_count: int
    pending_code: Optional[str]
    last_was_cached: bool
    last_similarity: Optional[float]


def run_code(code: str, local_vars: dict):
    """
    Runs the model's code INSIDE A SANDBOX. Captures printed output AND
    any matplotlib figures it created. Returns (text_output, images).

    Two safety layers before/during execution:
    1. Static check rejects obviously dangerous code before running it.
    2. exec() runs with restricted builtins/imports (see sandbox.py) —
       no file access, no eval/exec, no os/sys/subprocess.
    """
    safety_issue = sandbox.is_code_safe(code)
    if safety_issue:
        return (f"BLOCKED before running: {safety_issue}. "
                f"This sandbox restricts file/network/system access for safety."), []

    buf = io.StringIO()
    images = []
    try:
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(buf):
                exec_namespace = sandbox.get_sandbox_globals()
                exec_namespace.update(local_vars)
                exec(code, exec_namespace)

        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight")
            img_buf.seek(0)
            images.append(base64.b64encode(img_buf.read()).decode("utf-8"))
        plt.close("all")  # always clean up, even on success
    except Exception as e:
        plt.close("all")
        return f"ERROR while executing code:\n{e}", []

    output = buf.getvalue() or "(code ran, no printed output)"

    # Surface warnings explicitly — these often indicate a SILENT failure
    # (e.g. pandas chained-assignment + inplace=True can look successful
    # but not actually modify the dataframe). The model needs to see this
    # to know its "fix" might not have actually worked.
    if captured_warnings:
        warning_lines = [f"{w.category.__name__}: {w.message}" for w in captured_warnings]
        output += "\n\n[PYTHON WARNINGS — your last operation may not have worked as intended]\n"
        output += "\n".join(warning_lines)

    return output, images


def think_node(state: AgentState) -> dict:
    """THINK: ask the model what to do next, given everything observed so far.
    Checks the semantic cache first — if a sufficiently similar conversation
    state has been seen before, reuse that response instead of calling the API."""
    # Fingerprint the dataframe AND the current system prompt version.
    # Dataset shape/columns prevents cross-dataset collisions (see earlier
    # fix). Prompt version prevents an old cached answer from a PREVIOUS
    # version of SYSTEM_PROMPT being served after we've changed it.
    fingerprint = f"{tuple(state['df'].columns)}|{state['df'].shape}|{PROMPT_VERSION}"

    cached_reply, similarity = cache.get_cached_response(state["messages"], fingerprint)
    was_cached = cached_reply is not None
    similarity_score = round(similarity, 4) if similarity is not None else None

    if cached_reply is not None:
        reply = cached_reply
    else:
        response = client.chat.completions.create(
            model=MODEL, max_tokens=500, messages=state["messages"],
        )
        reply = response.choices[0].message.content
        cache.set_cached_response(state["messages"], reply, fingerprint)

    new_messages = state["messages"] + [{"role": "assistant", "content": reply}]

    # Look for a standalone "DONE" line ANYWHERE in the reply — the model
    # sometimes buries it after a block of reasoning comments instead of
    # putting it first, which broke a naive startswith() check.
    lines = reply.split("\n")
    done_idx = next((i for i, line in enumerate(lines) if line.strip() == "DONE"), None)

    if done_idx is not None or state["step_count"] >= MAX_STEPS:
        if done_idx is not None:
            summary_lines = [l for l in lines[done_idx:] if not l.strip().startswith("```")]
            summary = "\n".join(summary_lines).strip()
        else:
            summary = "Stopped: max steps reached"
        return {**state, "messages": new_messages, "finished": True, "summary": summary,
                "last_was_cached": was_cached, "last_similarity": similarity_score}

    if "```" in reply:
        code = reply.split("```")[1].replace("python", "", 1).strip()
    else:
        code = reply.strip()

    return {**state, "messages": new_messages, "pending_code": code,
            "last_was_cached": was_cached, "last_similarity": similarity_score}


def act_node(state: AgentState) -> dict:
    """ACT + OBSERVE: run the code the model just chose, record the result."""
    code = state["pending_code"]
    result_text, images = run_code(code, {"df": state["df"], "pd": pd})

    new_steps = state["steps"] + [
        {"step": state["step_count"] + 1, "code": code, "result": result_text,
         "images": images, "from_cache": state["last_was_cached"],
         "cache_similarity": state["last_similarity"]}
    ]

    # tell the model a chart was made, but never send raw image bytes back —
    # the model can't "see" it, and it would waste huge amounts of token budget
    feedback = result_text
    if images:
        feedback += f"\n\n[{len(images)} chart(s) generated and saved for the user to view]"

    new_messages = state["messages"] + [
        {"role": "user", "content": f"Output of your code:\n{feedback}\n\n"
                                     f"Given this, what's the most useful next step?"}
    ]

    return {
        **state,
        "steps": new_steps,
        "messages": new_messages,
        "step_count": state["step_count"] + 1,
        "pending_code": None,
    }


def route_after_think(state: AgentState) -> str:
    return "end" if state["finished"] else "act"


# --- Build the graph ---
_graph = StateGraph(AgentState)
_graph.add_node("think", think_node)
_graph.add_node("act", act_node)
_graph.set_entry_point("think")
_graph.add_conditional_edges("think", route_after_think, {"end": END, "act": "act"})
_graph.add_edge("act", "think")
compiled_graph = _graph.compile()


def run_agent_graph(df: pd.DataFrame, messages: Optional[List[dict]] = None) -> dict:
    """
    Runs the THINK/ACT/OBSERVE loop starting from the given message history.
    If messages is None, this is a brand-new analysis (builds the first prompt).
    If messages is provided, this CONTINUES an existing conversation — e.g. a
    follow-up question — and the model can see everything found previously.
    """
    if messages is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"The dataframe `df` has columns: {list(df.columns)} "
                                         f"and shape {df.shape}. What's your first analysis step?"},
        ]

    initial_state: AgentState = {
        "df": df,
        "messages": messages,
        "steps": [],          # only NEW steps from this call — caller merges with history
        "finished": False,
        "summary": None,
        "step_count": 0,      # fresh step budget for this call
        "pending_code": None,
        "last_was_cached": False,
        "last_similarity": None,
    }

    final_state = compiled_graph.invoke(initial_state, config={"recursion_limit": 50})

    return {
        "steps": final_state["steps"],
        "messages": final_state["messages"],   # full updated history, for the caller to persist
        "finished": final_state["finished"],
        "summary": final_state["summary"],
        "stopped_reason": "model said DONE" if final_state["finished"] and final_state["summary"]
                           and "Stopped" not in final_state["summary"] else "max steps reached",
    }