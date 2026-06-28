"""
test_sandbox.py — manual proof that the sandbox actually blocks dangerous
code while still letting legitimate pandas/numpy/matplotlib work happen.

Run with: python test_sandbox.py
"""

import pandas as pd
from graph import run_code

df = pd.DataFrame({"a": [1, 2, 3]})

dangerous_tests = [
    "import os\nos.system('echo hacked')",
    "open('test_sandbox.py').read()",
    "eval('1+1')",
    "__import__('subprocess').run(['ls'])",
    "import shutil\nshutil.rmtree('/')",
]

safe_tests = [
    "print(df.describe())",
    "import numpy as np\nprint(np.mean(df['a']))",
]

print("=== Dangerous code (every one of these should be BLOCKED) ===")
for code in dangerous_tests:
    result, _ = run_code(code, {"df": df, "pd": pd})
    blocked = "BLOCKED" in result or "ImportError" in result or "ERROR" in result
    status = "BLOCKED (correct)" if blocked else "!!! NOT BLOCKED — this is a bug !!!"
    print(f"\n[{status}]")
    print(f"  code:   {code.splitlines()[0]}")
    print(f"  result: {result[:150]}")

print("\n\n=== Safe code (should run completely normally) ===")
for code in safe_tests:
    result, _ = run_code(code, {"df": df, "pd": pd})
    print(f"\ncode:   {code.splitlines()[0]}")
    print(f"result: {result[:200]}")
