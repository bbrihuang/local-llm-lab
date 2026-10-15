"""The shared part of the test: the tools every agent gets, the questions, and the scoring.

Every framework gets exactly the same three tools (plain Python functions) and the same
instructions. Each framework only wraps them in its own way.
"""
import ast
import operator
import os
import re
import threading

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Where the agents send their requests: the counting proxy in bench.py, in front of llama-server.
BASE_URL = os.environ.get("AGENT_BASE_URL", "http://127.0.0.1:8090/v1")
MODEL = "qwen3-8b"   # llama-server serves whatever model it loaded under any name
API_KEY = "none"     # llama-server doesn't check it, but the client libraries want one
MAX_STEPS = 10

INSTRUCTIONS = (
    "You are a helpful assistant. You can list and read the files in the data folder and "
    "use a calculator. Use the calculator for any arithmetic instead of doing it in your head. "
    "When you know the answer, reply with a short final answer."
)

# Counts tool calls, whichever framework makes them.
_lock = threading.Lock()
tool_calls = {"count": 0}


def _count():
    with _lock:
        tool_calls["count"] += 1


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / ** % ( ) are allowed")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, for example "(150.9 / 64.6) * 100".

    Args:
        expression: The expression to evaluate. Use numbers, + - * / ** % and parentheses.
    """
    _count()
    try:
        value = _eval(ast.parse(expression.replace(",", "").replace("^", "**"), mode="eval").body)
        return str(round(value, 6))
    except Exception as e:
        return f"Error: {e}"


def list_files() -> str:
    """List the files in the data folder."""
    _count()
    return "\n".join(sorted(os.listdir(DATA_DIR)))


def read_file(name: str) -> str:
    """Read one file from the data folder and return its text.

    Args:
        name: The file name, for example "gpus.txt".
    """
    _count()
    path = os.path.join(DATA_DIR, os.path.basename(name.strip()))
    if not os.path.isfile(path):
        return f"Error: no file named {name!r}. Use list_files to see the files."
    with open(path, encoding="utf-8") as f:
        return f.read()


TOOLS = [calculator, list_files, read_file]

# (id, what it tests, question, expected answer)
# A number is correct if the answer contains it (to the given decimals); text is matched
# case-insensitively.
TASKS = [
    ("t01", "calculator", "What is 17.5% of 2,348?", 410.9),
    ("t02", "calculator", "What is 2 to the power of 20, minus 1,000,000?", 48576),
    ("t03", "no tools needed", "What is the capital of France?", "paris"),
    ("t04", "list files", "How many files are in the data folder?", 3),
    ("t05", "read a file", "In quantization.csv, which format generated tokens the fastest?", "q2_k"),
    ("t06", "read a file", "Which file says how much memory the Tesla T4 has, and how many GB is it?", 16),
    ("t07", "read + calculate",
     "Using quantization.csv, how many times faster than F16 does Q4_K_M generate tokens? "
     "Round to 2 decimals.", 2.34),
    ("t08", "read + calculate",
     "Using gpus.txt, what is the most tokens per second the RTX 5070 could generate with an "
     "8.05 GB model? Round to 1 decimal.", 83.5),
    ("t09", "read + calculate",
     "Using engines.csv, at 32 users, how many percent higher was llama.cpp's total throughput "
     "than vLLM's on the Tesla T4? Round to 1 decimal.", 19.0),
    ("t10", "read + filter + calculate",
     "Using quantization.csv, add up size_gib for every row whose kl_divergence is below 0.05. "
     "Round to 2 decimals.", 17.5),
]


def is_correct(answer, expected):
    text = str(answer or "").lower()
    if isinstance(expected, str):
        return expected in text
    numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]
    return any(abs(n - expected) < 0.051 for n in numbers)
