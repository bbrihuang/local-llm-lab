"""Run every agent framework on the same questions with the same local model, and record
whether it got the answer right, how long it took, and how much work it made the model do.

Run from the 07.5-agent-frameworks folder:  ..\\.venv\\Scripts\\python bench.py

It starts llama-server with Qwen3-8B, plus a small "counting proxy" between the agents and
the server. Every request any framework sends goes through the proxy, which
  - counts the requests and the tokens read and written, whatever the framework, and
  - sets the same sampling settings on every request (some frameworks send their own
    temperature and some send none), turns Qwen3's thinking off so runs stay short, and
    caps every reply at 1,024 tokens so a model that starts rambling can't run for minutes.
"""
import argparse
import csv
import http.server
import importlib
import json
import os
import socketserver
import statistics
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
import tasks  # noqa: E402  (after chdir so it's found when run from elsewhere)

FRAMEWORKS = ["plain", "langgraph", "smolagents", "openai_agents", "pydantic_ai"]

ap = argparse.ArgumentParser()
ap.add_argument("--server", default=r"C:\Users\brian\tools\llama.cpp\llama-server.exe")
ap.add_argument("--model", default=r"C:\Users\brian\models\Qwen3-8B-Q4_K_M.gguf")
ap.add_argument("--no-server", action="store_true", help="use a llama-server that is already running")
ap.add_argument("--upstream", default="http://127.0.0.1:8080")
ap.add_argument("--frameworks", default=",".join(FRAMEWORKS))
ap.add_argument("--tasks", default="", help="comma-separated task ids; default all")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--ctx", type=int, default=8192, help="context size for llama-server")
ap.add_argument("--cuda-graphs", action="store_true",
                help="leave llama.cpp's CUDA graphs on (off by default here; see README)")
ap.add_argument("--out", default="results/runs.csv")
args = ap.parse_args()

SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,  # Qwen3's advice for non-thinking mode
            "chat_template_kwargs": {"enable_thinking": False}}
MAX_REPLY_TOKENS = 1024
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# ---------------------------------------------------------------- counting proxy
stats_lock = threading.Lock()
stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def record(usage):
    with stats_lock:
        stats["calls"] += 1
        stats["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
        stats["completion_tokens"] += usage.get("completion_tokens", 0) or 0


class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def forward(self, body=None):
        req = urllib.request.Request(args.upstream + self.path, data=body, method=self.command,
                                     headers={"Content-Type": "application/json"})
        try:
            return DIRECT.open(req, timeout=600)
        except urllib.error.HTTPError as e:
            return e

    def reply(self, resp, data):
        self.send_response(resp.status if hasattr(resp, "status") else resp.code)
        self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        resp = self.forward()
        self.reply(resp, resp.read())

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.endswith("/chat/completions"):
            payload = json.loads(body)
            payload.update(SAMPLING)
            payload.pop("max_completion_tokens", None)
            payload["max_tokens"] = MAX_REPLY_TOKENS
            if payload.get("stream"):
                payload["stream_options"] = {"include_usage": True}
            body = json.dumps(payload).encode()
        resp = self.forward(body)
        if not self.path.endswith("/chat/completions") or getattr(resp, "code", 200) >= 400:
            self.reply(resp, resp.read())
            return
        if "text/event-stream" in resp.headers.get("Content-Type", ""):
            # Pass the stream through chunk by chunk, and pick the usage out of it.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            usage = {}
            for line in resp:
                self.wfile.write(line)
                self.wfile.flush()
                text = line.decode("utf-8", "replace").strip()
                if text.startswith("data:") and '"usage"' in text:
                    try:
                        usage = json.loads(text[5:]).get("usage") or usage
                    except ValueError:
                        pass
            record(usage)
        else:
            data = resp.read()
            record(json.loads(data).get("usage") or {})
            self.reply(resp, data)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def health():
    try:
        with DIRECT.open(args.upstream + "/health", timeout=2) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


# ---------------------------------------------------------------- main
server_proc = None
if not args.no_server:
    if health() is not None:
        raise SystemExit("Something is already using port 8080. Close other llama-server windows, "
                         "or run: taskkill /F /IM llama-server.exe")
    for path in (args.server, args.model):
        if not os.path.exists(path):
            raise SystemExit(f"Not found: {path}")
    os.makedirs("logs", exist_ok=True)
    cmd = [args.server, "-m", args.model, "-ngl", "99", "-c", str(args.ctx), "-np", "1",
           "--jinja", "--port", "8080"]
    print("Starting:", " ".join(cmd))
    log = open("logs/server.log", "w", encoding="utf-8")
    env = dict(os.environ)
    if not args.cuda_graphs:
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    server_proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while health() != 200:
        if server_proc.poll() is not None:
            raise SystemExit("llama-server stopped while loading. See logs/server.log.")
        if time.time() - t0 > 600:
            raise SystemExit("llama-server didn't finish loading in 10 minutes.")
        time.sleep(2)
    print(f"Server ready after {time.time() - t0:.0f} s")
elif health() != 200:
    raise SystemExit(f"No llama-server answering at {args.upstream}")

proxy = Server(("127.0.0.1", 8090), Proxy)
threading.Thread(target=proxy.serve_forever, daemon=True).start()

FIELDS = ["framework", "task", "kind", "repeat", "correct", "seconds", "llm_calls",
          "prompt_tokens", "completion_tokens", "tool_calls", "error", "answer"]
chosen = [t for t in tasks.TASKS if not args.tasks or t[0] in args.tasks.split(",")]
names = args.frameworks.split(",")
rows = []
try:
    modules = {}
    for name in names:
        print(f"Loading {name} ...")
        modules[name] = importlib.import_module(f"agent_{name}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Re-running a framework replaces its earlier rows instead of adding to them.
    if os.path.exists(args.out):
        with open(args.out, newline="", encoding="utf-8") as f:
            keep = [r for r in csv.DictReader(f) if r["framework"] not in names]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(keep)
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for rep in range(1, args.repeats + 1):
            for task_id, kind, question, expected in chosen:
                for name in names:  # frameworks take turns, so any drift hits all of them
                    with stats_lock:
                        stats.update(calls=0, prompt_tokens=0, completion_tokens=0)
                    tasks.tool_calls["count"] = 0
                    t0 = time.perf_counter()
                    answer, error = "", ""
                    try:
                        answer = modules[name].run(question)
                    except Exception as e:
                        error = f"{type(e).__name__}: {e}"[:200]
                        if os.environ.get("BENCH_DEBUG"):
                            traceback.print_exc()
                    secs = time.perf_counter() - t0
                    ok = not error and tasks.is_correct(answer, expected)
                    row = dict(framework=name, task=task_id, kind=kind, repeat=rep, correct=int(ok),
                               seconds=round(secs, 2), llm_calls=stats["calls"],
                               prompt_tokens=stats["prompt_tokens"],
                               completion_tokens=stats["completion_tokens"],
                               tool_calls=tasks.tool_calls["count"], error=error,
                               answer=" ".join(str(answer or "").split())[:300])
                    rows.append(row)
                    w.writerow(row)
                    f.flush()
                    mark = "ok " if ok else ("ERR" if error else "no ")
                    print(f"[{rep}/{args.repeats}] {task_id} {name:<14} {mark} {secs:5.1f} s  "
                          f"{row['llm_calls']} calls  {row['tool_calls']} tools  "
                          f"{(error or row['answer'])[:60]}")
finally:
    proxy.shutdown()
    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("Stopped llama-server")

print(f"\n{'framework':<14} {'correct':>8} {'errors':>7} {'median s':>9} {'calls':>6} {'tokens in':>10} {'tokens out':>11}")
for name in names:
    r = [x for x in rows if x["framework"] == name]
    if not r:
        continue
    print(f"{name:<14} {sum(x['correct'] for x in r):>4}/{len(r):<3} {sum(1 for x in r if x['error']):>7} "
          f"{statistics.median(x['seconds'] for x in r):>9.1f} "
          f"{statistics.mean(x['llm_calls'] for x in r):>6.1f} "
          f"{statistics.mean(x['prompt_tokens'] for x in r):>10.0f} "
          f"{statistics.mean(x['completion_tokens'] for x in r):>11.0f}")
print(f"Saved {args.out}")
