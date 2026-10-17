"""Task 10: test Qwen3.5-4B's tool calling with 15 requests and three tools.

Run from the 10-tool-calling folder:
    ..\\.venv\\Scripts\\python run_tools.py --label q4_k_m --model C:\\Users\\brian\\models\\Qwen3.5-4B-Q4_K_M.gguf
    ..\\.venv\\Scripts\\python run_tools.py --label q2_k --model C:\\Users\\brian\\models\\Qwen3.5-4B-Q2_K.gguf

It starts llama-server with --jinja (needed for tool calling), asks each request --repeats times
(sampling is random, so one try is not enough), scores the FIRST reply of each try, and stops the server.
Scoring a try:
  correct   exactly the expected tool calls, with right arguments, and no extra calls
            (for the 'no tool' requests: no tool call at all)
  right_tool  the tool names match, even if an argument is wrong
  valid_json  every call's arguments parsed as JSON
Add --url http://127.0.0.1:8080 to use a server you already started instead of starting one.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter

from cases import CASES, TOOLS

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True, help="name for this run, e.g. q4_k_m")
ap.add_argument("--model", default=r"C:\Users\brian\models\Qwen3.5-4B-Q4_K_M.gguf")
ap.add_argument("--server", default=r"C:\Users\brian\tools\llama.cpp\llama-server.exe")
ap.add_argument("--thinking", action="store_true", help="turn thinking on (default is off)")
ap.add_argument("--repeats", type=int, default=5, help="tries per request")
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--url", default="", help="use an already running server")
ap.add_argument("--limit", type=int, default=0, help="only the first N requests (quick test)")
args = ap.parse_args()

mode = "thinking" if args.thinking else "instant"
sampling = ({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 14000}
            if args.thinking else
            {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 1024})
base = args.url or f"http://127.0.0.1:{args.port}"
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def health():
    try:
        with DIRECT.open(base + "/health", timeout=2) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


def ask(prompt):
    body = {"messages": [{"role": "user", "content": prompt}], "tools": TOOLS, **sampling}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with DIRECT.open(req, timeout=600) as r:
        data = json.load(r)
    return data["choices"][0], time.time() - t0, data.get("usage", {})


def score(case, message):
    calls, json_ok = [], True
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        raw = fn.get("arguments")
        try:
            a = json.loads(raw) if isinstance(raw, str) else (raw or {})
            if not isinstance(a, dict):
                raise ValueError
        except Exception:
            a, json_ok = {}, False
        calls.append((fn.get("name"), a))
    want = case["expect"]
    right_tool = Counter(n for n, _ in calls) == Counter(n for n, _ in want)
    correct = False
    if right_tool and json_ok:
        left = list(calls)
        correct = True
        for name, check in want:  # match each expected call to a different actual call
            hit = next((c for c in left if c[0] == name and check(c[1])), None)
            if hit is None:
                correct = False
                break
            left.remove(hit)
    return {"calls": [{"name": n, "args": a} for n, a in calls], "valid_json": json_ok,
            "right_tool": right_tool, "correct": correct}


server = None
if not args.url:
    if health() is not None:
        raise SystemExit(f"Something is already using port {args.port}. Close other llama-server windows, "
                         "or run: taskkill /F /IM llama-server.exe")
    for p in (args.server, args.model):
        if not os.path.exists(p):
            raise SystemExit(f"Not found: {p}")
    os.makedirs("logs", exist_ok=True)
    cmd = [args.server, "-m", args.model, "-ngl", "99", "-c", str(args.ctx), "--jinja",
           "--chat-template-kwargs", json.dumps({"enable_thinking": args.thinking}),
           "--host", "127.0.0.1", "--port", str(args.port)]
    print("Starting:", " ".join(cmd))
    log = open(f"logs/server_{args.label}_{mode}.log", "w", encoding="utf-8")
    env = dict(os.environ, GGML_CUDA_DISABLE_GRAPHS="1")
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

try:
    t0 = time.time()
    while health() != 200:
        if server and server.poll() is not None:
            raise SystemExit("llama-server stopped while loading. See the log in the logs folder.")
        if time.time() - t0 > 600:
            raise SystemExit("llama-server didn't finish loading in 10 minutes.")
        time.sleep(2)
    print(f"Server ready after {time.time() - t0:.0f} s")

    cases = CASES[:args.limit] if args.limit else CASES
    os.makedirs("results", exist_ok=True)
    out_path = f"results/tools_{args.label}_{mode}.jsonl"
    total = right = fine = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for case in cases:
            res = []
            for k in range(args.repeats):
                choice, secs, usage = ask(case["prompt"])
                msg = choice["message"]
                s = score(case, msg)
                res.append(s["correct"])
                total += 1
                right += s["correct"]
                fine += s["right_tool"]
                out.write(json.dumps({"label": args.label, "mode": mode, "case": case["id"], "kind": case["kind"],
                                      "prompt": case["prompt"], "try": k, "seconds": round(secs, 2),
                                      "completion_tokens": usage.get("completion_tokens"),
                                      "finish_reason": choice.get("finish_reason"),
                                      "reply_text": (msg.get("content") or "")[:500], **s}) + "\n")
                out.flush()
            print(f"case {case['id']:>2} [{case['kind']:<6}] {sum(res)}/{args.repeats} correct  {case['prompt'][:50]}")
    print(f"\n{args.label} ({mode}): {right}/{total} correct ({100 * right / total:.1f}%), "
          f"right tool {fine}/{total} ({100 * fine / total:.1f}%)")
    print("Saved", out_path)
finally:
    if server:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        print("Stopped llama-server")
