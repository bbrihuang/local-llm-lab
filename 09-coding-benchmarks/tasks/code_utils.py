"""HumanEval for chat models: prompt, code extraction, and running the tests.

The model gets the function header and docstring and must reply with the finished function
in a ```python block. The reply's code is added after the original header, then HumanEval's
own test (check(...)) is run in a separate Python process with a 10 second limit.

This runs model-written code. Run it inside the Docker container (run_humaneval.py does).
"""
import os
import re
import subprocess
import sys
import tempfile

INSTRUCTION = ("Complete the following Python function. Reply with the complete function "
               "(including any imports it needs) in a single ```python code block.")
TIMEOUT = 10


def he_text(doc):
    return f"{INSTRUCTION}\n\n```python\n{doc['prompt'].strip()}\n```"


def extract_code(reply, entry_point):
    """The python block that defines the function (else the longest block); None if no block."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", reply, flags=re.S | re.I)
    if not blocks:
        # A block that was cut off before its closing fence still counts.
        m = re.search(r"```(?:python|py)?[ \t]*\n(.*)$", reply, flags=re.S | re.I)
        blocks = [m.group(1)] if m else []
    if not blocks:
        return None
    for b in blocks:
        if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", b):
            return b
    return max(blocks, key=len)


def build_program(doc, code):
    # The original header first (it can hold imports and helper functions the test needs),
    # then the model's code, which redefines the function.
    return f"{doc['prompt']}\n    pass\n\n\n{code}\n\n\n{doc['test']}\n\ncheck({doc['entry_point']})\n"


def run_program(program):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, path], cwd=d, capture_output=True,
                               timeout=TIMEOUT, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


def he_results(doc, results):
    code = extract_code(results[0], doc["entry_point"])
    if code is None:
        return {"pass_at_1": 0, "no_code_block": 1}
    return {"pass_at_1": int(run_program(build_program(doc, code))), "no_code_block": 0}
