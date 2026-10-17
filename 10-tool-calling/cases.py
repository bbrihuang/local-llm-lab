"""The three tools and the 15 test requests for Task 10.

Nothing here really runs: the model only decides WHICH tool to call and with WHAT arguments.
We never execute the weather lookup or the file search. We only read the model's choice and score it.
"""
import ast
import operator

TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression such as '12 * (3 + 4)'. Use it for any calculation.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "The arithmetic expression to evaluate"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                     "description": "Temperature unit. Default celsius."}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_files",
        "description": "Search the user's files by name or content.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Words to search for"},
            "folder": {"type": "string", "description": "Optional folder to search in, e.g. 'Documents'"}},
            "required": ["query"]}}},
]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}


def calc(expr):
    """Safely evaluate an arithmetic expression. Returns None if it isn't plain arithmetic."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError
    try:
        text = str(expr).replace(",", "").replace("$", "")
        return ev(ast.parse(text.strip(), mode="eval"))
    except Exception:
        return None


def is_num(expr, target):
    v = calc(expr)
    return v is not None and abs(v - target) < 1e-6


def _has(s, word):
    return isinstance(s, str) and word.lower() in s.lower()


# Each check takes the arguments dict and returns True if the arguments are right.
def calc_is(target):
    return lambda a: is_num(a.get("expression"), target)


def weather(city, unit=None):
    def f(a):
        ok = _has(a.get("city"), city)
        if unit:
            ok = ok and str(a.get("unit", "")).lower() == unit
        else:
            ok = ok and str(a.get("unit", "celsius")).lower() == "celsius"
        return ok
    return f


def search(word, folder=None):
    def f(a):
        ok = _has(a.get("query"), word)
        if folder:
            ok = ok and _has(a.get("folder"), folder)
        return ok
    return f


# expect = list of (tool name, argument check). An empty list means NO tool should be called.
CASES = [
    # --- one tool call ---
    {"id": 1, "kind": "single", "prompt": "What is 1847 * 23?",
     "expect": [("calculator", calc_is(42481))]},
    {"id": 2, "kind": "single", "prompt": "What is 15% of 2,480?",
     "expect": [("calculator", calc_is(372))]},
    {"id": 3, "kind": "single", "prompt": "What's the weather in Paris right now?",
     "expect": [("get_weather", weather("Paris"))]},
    {"id": 4, "kind": "single", "prompt": "How hot is it in Austin right now? Give me the answer in Fahrenheit.",
     "expect": [("get_weather", weather("Austin", "fahrenheit"))]},
    {"id": 5, "kind": "single", "prompt": "Find my file called budget_2025.xlsx.",
     "expect": [("search_files", search("budget_2025"))]},
    {"id": 6, "kind": "single", "prompt": "Look in my Documents folder for anything about the lease.",
     "expect": [("search_files", search("lease", "Documents"))]},
    {"id": 7, "kind": "single", "prompt": "I bought 3 notebooks at $4.50 each and a pen for $1.25. What's the total?",
     "expect": [("calculator", calc_is(14.75))]},
    # --- several calls in one go ---
    {"id": 8, "kind": "multi", "prompt": "What's the weather in Tokyo and in London?",
     "expect": [("get_weather", weather("Tokyo")), ("get_weather", weather("London"))]},
    {"id": 9, "kind": "multi", "prompt": "What is 12 * 12, and what's the weather in Boston?",
     "expect": [("calculator", calc_is(144)), ("get_weather", weather("Boston"))]},
    {"id": 10, "kind": "multi", "prompt": "Search my files for 'invoice' and also for 'receipt'.",
     "expect": [("search_files", search("invoice")), ("search_files", search("receipt"))]},
    # --- no tool should be used ---
    {"id": 11, "kind": "none", "prompt": "What is the capital of France?", "expect": []},
    {"id": 12, "kind": "none", "prompt": "Explain in one sentence what a calculator is.", "expect": []},
    {"id": 13, "kind": "none", "prompt": "Is it usually warmer in Spain than in Norway in July?", "expect": []},
    {"id": 14, "kind": "none", "prompt": "What's the weather like?", "expect": [],
     "note": "no city given, so the right move is to ask which city"},
    {"id": 15, "kind": "none", "prompt": "Tell me a short story about a filing cabinet.", "expect": []},
]
