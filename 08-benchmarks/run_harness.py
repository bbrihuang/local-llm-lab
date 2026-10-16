"""Run lm-evaluation-harness on Qwen3.5-4B (Task 3's Q4_K_M file) for ARC-Challenge and MMLU-Pro.

Run from the 08-benchmarks folder:
    ..\\.venv\\Scripts\\python run_harness.py --mode instant
    ..\\.venv\\Scripts\\python run_harness.py --mode thinking

It starts llama-server, runs the harness against it through the OpenAI-style chat endpoint,
then stops the server. The two tasks live in tasks/ (200 ARC-Challenge questions, and
15 questions from each of MMLU-Pro's 14 subjects = 210).

  instant   thinking off. Fast (minutes).
  thinking  thinking on, as in Qwen's official scores. Much slower: each answer can be
            thousands of tokens of reasoning.
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Qwen's recommended sampling for each mode (from the Qwen3.5-4B model card).
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
ap.add_argument("--tasks", default="arc_challenge_qwen,mmlu_pro_qwen")
ap.add_argument("--users", type=int, default=4, help="questions answered at the same time")
ap.add_argument("--ctx", type=int, default=16384, help="context per question")
ap.add_argument("--limit", default="", help="only run this many questions per task (for a quick test)")
ap.add_argument("--port", type=int, default=8080)
args = ap.parse_args()
mode = MODES[args.mode]
url = f"http://127.0.0.1:{args.port}"
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def health():
    try:
        with DIRECT.open(url + "/health", timeout=2) as r:
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

os.makedirs("logs", exist_ok=True)
cmd = [args.server, "-m", args.model, "-ngl", "99", "-c", str(args.ctx * args.users),
       "-np", str(args.users), "--jinja", "--chat-template-kwargs", mode["template"],
       "--port", str(args.port)]
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

    harness = [sys.executable, "-m", "lm_eval", "run",
               "--model", "local-chat-completions",
               "--model_args", f"model=qwen3.5-4b,base_url={url}/v1/chat/completions,"
                               f"num_concurrent={args.users},max_retries=3,timeout=3600",
               "--tasks", args.tasks, "--include_path", "tasks", "--apply_chat_template",
               "--gen_kwargs", mode["gen"],
               "--output_path", f"results/harness/{args.mode}", "--log_samples"]
    if args.limit:
        harness += ["--limit", args.limit]
    print("Running:", " ".join(harness))
    t0 = time.time()
    code = subprocess.call(harness, env=dict(os.environ, PYTHONUTF8="1"))
    print(f"Harness finished in {(time.time() - t0) / 60:.1f} minutes (exit code {code})")
finally:
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
    print("Stopped llama-server")
