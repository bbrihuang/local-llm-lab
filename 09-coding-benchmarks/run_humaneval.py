"""Run HumanEval on Qwen3.5-4B (Task 3's Q4_K_M file) with lm-evaluation-harness.

Run from the 09-coding-benchmarks folder:
    ..\\.venv\\Scripts\\python run_humaneval.py --mode instant
    ..\\.venv\\Scripts\\python run_humaneval.py --mode thinking

HumanEval runs code the model wrote. So the harness runs inside a throwaway Docker container
(python:3.12-slim, deleted when it finishes) and only the results folder is shared with it.
llama-server stays on your PC and the container talks to it at host.docker.internal.

  instant   thinking off. Fast.
  thinking  thinking on, Qwen3.5's default mode. Slower: thousands of tokens of reasoning each.

Add --limit 5 for a quick test on 5 problems. Add --no-docker to run on the PC itself
(not recommended: the model's code would run as you).
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

MODES = {
    "instant": {"template": '{"enable_thinking": false}',
                "gen": "temperature=0.7,top_p=0.8,top_k=20,presence_penalty=1.5,max_gen_toks=4096"},
    "thinking": {"template": '{"enable_thinking": true}',
                 "gen": "temperature=1.0,top_p=0.95,top_k=20,presence_penalty=1.5,max_gen_toks=14000"},
}

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=MODES, required=True)
ap.add_argument("--server", default=r"C:\Users\brian\tools\llama.cpp\llama-server.exe")
ap.add_argument("--model", default=r"C:\Users\brian\models\Qwen3.5-4B-Q4_K_M.gguf")
ap.add_argument("--users", type=int, default=4, help="problems answered at the same time")
ap.add_argument("--ctx", type=int, default=16384, help="context per problem")
ap.add_argument("--limit", default="", help="only run this many problems (for a quick test)")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--no-docker", action="store_true")
args = ap.parse_args()
mode = MODES[args.mode]
local_url = f"http://127.0.0.1:{args.port}"
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def health():
    try:
        with DIRECT.open(local_url + "/health", timeout=2) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


if health() is not None:
    raise SystemExit(f"Something is already using port {args.port}. Close other llama-server windows, "
                     "or run: taskkill /F /IM llama-server.exe")
for path in (args.server, args.model):
    if not os.path.exists(path):
        raise SystemExit(f"Not found: {path}")
if not args.no_docker:
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=60)
    except Exception:
        raise SystemExit("Docker isn't running. Start Docker Desktop, wait until it says "
                         "'Engine running', and try again.")

os.makedirs("logs", exist_ok=True)
host = "127.0.0.1" if args.no_docker else "0.0.0.0"  # the container reaches the PC from outside
cmd = [args.server, "-m", args.model, "-ngl", "99", "-c", str(args.ctx * args.users),
       "-np", str(args.users), "--jinja", "--chat-template-kwargs", mode["template"],
       "--host", host, "--port", str(args.port)]
print("Starting:", " ".join(cmd))
log = open(f"logs/server_{args.mode}.log", "w", encoding="utf-8")
env = dict(os.environ, GGML_CUDA_DISABLE_GRAPHS="1")  # see Task 7.5: CUDA graphs froze the PC once
server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
try:
    t0 = time.time()
    while health() != 200:
        if server.poll() is not None:
            raise SystemExit(f"llama-server stopped while loading. See logs/server_{args.mode}.log.")
        if time.time() - t0 > 600:
            raise SystemExit("llama-server didn't finish loading in 10 minutes.")
        time.sleep(2)
    print(f"Server ready after {time.time() - t0:.0f} s")

    base = local_url if args.no_docker else f"http://host.docker.internal:{args.port}"
    harness = ["python", "-m", "lm_eval", "run",
               "--model", "local-chat-completions",
               "--model_args", f"model=qwen3.5-4b,base_url={base}/v1/chat/completions,"
                               f"num_concurrent={args.users},max_retries=3,timeout=3600",
               "--tasks", "humaneval_qwen", "--include_path", "tasks", "--apply_chat_template",
               "--gen_kwargs", mode["gen"],
               "--output_path", f"results/harness/{args.mode}", "--log_samples"]
    if args.limit:
        harness += ["--limit", args.limit]
    if args.no_docker:
        harness[0] = sys.executable
        run = harness
    else:
        inner = "pip install -q 'lm-eval[api]==0.4.13' && " + " ".join(
            f"'{a}'" if " " in a or "," in a else a for a in harness)
        run = ["docker", "run", "--rm", "--add-host", "host.docker.internal:host-gateway",
               "-e", "PYTHONUTF8=1", "-v", f"{HERE}:/work", "-w", "/work",
               "python:3.12-slim", "bash", "-c", inner]
    print("Running:", " ".join(run))
    t0 = time.time()
    code = subprocess.call(run, env=dict(os.environ, PYTHONUTF8="1"))
    print(f"Harness finished in {(time.time() - t0) / 60:.1f} minutes (exit code {code})")
finally:
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
    print("Stopped llama-server")
