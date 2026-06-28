"""
sandbox.py — lightweight code-execution sandbox.

Honest scope: this is NOT bulletproof against a deliberately adversarial
payload (creative dunder-attribute chaining could still find a crack in
pure-Python restricted-builtins sandboxing). What this DOES protect well
against: the realistic failure mode here — an LLM trying to be helpful
that writes something that would touch the filesystem, network, or OS.
True hardened isolation against a hostile actor means process/container
isolation (Docker, gVisor) — a heavier upgrade, not this.

Two layers:
1. A static pattern check that rejects obviously dangerous code BEFORE
   it's ever executed.
2. A restricted globals dict for exec() — only a curated safe subset of
   builtins is available, and only an explicit allowlist of modules can
   be imported.
"""

import builtins as _builtins

# Modules the agent is allowed to import. Anything else raises ImportError.
ALLOWED_MODULES = {
    "pandas", "numpy", "matplotlib", "matplotlib.pyplot",
    "math", "statistics", "json", "re", "datetime", "itertools", "collections",
}

# Substrings that immediately block code before it's even run.
# Defense in depth on top of the restricted builtins below.
FORBIDDEN_PATTERNS = [
    "os.", "sys.", "subprocess", "socket", "shutil", "importlib",
    "eval(", "exec(", "compile(", "__import__", "open(",
    "globals(", "locals(", "getattr(", "setattr(", "delattr(",
    "__subclasses__", "__bases__", "__globals__", "__code__",
    "pickle", "ctypes", "requests", "urllib", "ftplib",
]


def is_code_safe(code: str):
    """Returns None if the code passes the static check, or a reason string if blocked."""
    lowered = code.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.lower() in lowered:
            return f"contains forbidden pattern: '{pattern}'"
    return None


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Replaces __import__ inside the sandbox — only allowlisted modules load."""
    root = name.split(".")[0]
    allowed_roots = {m.split(".")[0] for m in ALLOWED_MODULES}
    if root not in allowed_roots:
        raise ImportError(
            f"Import of '{name}' is not allowed in this sandbox. "
            f"Allowed modules: {', '.join(sorted(ALLOWED_MODULES))}"
        )
    return _builtins.__import__(name, globals, locals, fromlist, level)


# A curated, safe subset of Python's builtins — no file/network/process
# access, no eval/exec, no introspection helpers that could be abused.
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "format", "frozenset", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "object", "pow", "print",
    "range", "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "type", "zip", "True", "False", "None",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "StopIteration", "AttributeError", "ZeroDivisionError", "RuntimeError",
    "NotImplementedError", "ArithmeticError", "OverflowError",
    "__build_class__", "__name__",
]

SAFE_BUILTINS = {
    name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(_builtins, name)
}
SAFE_BUILTINS["__import__"] = _restricted_import


def get_sandbox_globals():
    """A fresh globals dict with restricted builtins, for use with exec()."""
    return {"__builtins__": SAFE_BUILTINS}
