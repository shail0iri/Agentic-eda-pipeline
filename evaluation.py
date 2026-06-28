"""
evaluation.py — automated test suite for the agent.

Approach: build synthetic datasets with a KNOWN, deliberately injected
issue (a specific column has N nulls, a specific value is an outlier,
specific rows are exact duplicates). Run the real agent against each one,
then check whether its own steps/summary actually reference that known
issue. This is a narrower claim than "the agent reasoned well" — it's
"the agent's own output demonstrates it noticed the planted issue" —
but it's checkable without a human grader or an extra paid LLM-judge call.

NOTE: running this suite makes REAL Groq API calls (one full agent run
per test case, each up to MAX_STEPS turns). Don't run it on every save —
treat it like a periodic sanity check, not a unit test you run constantly.
"""

import pandas as pd
from graph import run_agent_graph


def _build_test_cases():
    """Each case: a dataframe with a known planted issue, plus what
    evidence we'd expect to see if the agent actually noticed it."""
    cases = []

    # Case 1: missing values in a specific column
    df1 = pd.DataFrame({
        "age": [25, 30, None, 40, None, 35, 28, 50],
        "salary": [50000, 62000, 58000, 61000, 49000, 53000, 60000, 57000],
        "city": ["NY", "LA", "NY", "SF", "NY", "LA", "SF", "NY"],
    })
    cases.append({
        "name": "missing_values",
        "df": df1,
        "expected_column": "age",
        "expected_keywords": ["null", "missing", "isnull", "nan"],
    })

    # Case 2: an obvious numeric outlier
    df2 = pd.DataFrame({
        "age": [25, 30, 35, 40, 28, 33, 45, 38],
        "salary": [50000, 62000, 58000, 61000, 49000, 53000, 9999999, 57000],
        "city": ["NY", "LA", "NY", "SF", "NY", "LA", "SF", "NY"],
    })
    cases.append({
        "name": "outlier",
        "df": df2,
        "expected_column": "salary",
        "expected_keywords": ["outlier", "extreme", "anomal", "unusual"],
    })

    # Case 3: exact duplicate rows
    base = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "salary": [50000, 62000, 58000, 61000],
        "city": ["NY", "LA", "NY", "SF"],
    })
    df3 = pd.concat([base, base.iloc[[0, 1]]], ignore_index=True)
    cases.append({
        "name": "duplicate_rows",
        "df": df3,
        "expected_column": None,
        "expected_keywords": ["duplicate", "drop_duplicates"],
    })

    return cases


def _check_detection(steps, summary, expected_keywords, expected_column):
    """Scans every step's code + printed result, plus the final summary,
    for evidence the agent noticed the known planted issue."""
    haystack = (summary or "").lower()
    for step in steps:
        haystack += " " + step.get("code", "").lower()
        haystack += " " + step.get("result", "").lower()

    keyword_hit = any(kw.lower() in haystack for kw in expected_keywords)
    column_hit = (expected_column is None) or (expected_column.lower() in haystack)
    return keyword_hit and column_hit


def run_eval_suite():
    cases = _build_test_cases()
    results = []

    for case in cases:
        result = run_agent_graph(case["df"])
        detected = _check_detection(
            result["steps"], result["summary"],
            case["expected_keywords"], case["expected_column"],
        )
        results.append({
            "case": case["name"],
            "detected_issue": detected,
            "finished_cleanly": result["finished"],
            "num_steps": len(result["steps"]),
            "stopped_reason": result["stopped_reason"],
            "summary": result["summary"],
            "steps": result["steps"],  # full step-by-step trace, for diagnosing failures
        })

    passed = sum(1 for r in results if r["detected_issue"])
    total = len(results)

    return {
        "score": f"{passed}/{total}",
        "pass_rate": round(passed / total, 2) if total else 0,
        "results": results,
    }