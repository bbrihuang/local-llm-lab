"""Run the same speed test as the Colab notebook, with llama-server on this PC.

Starts llama-server with Qwen3-4B F16, waits until it has loaded, runs
engine_bench.py against it, then stops the server. Only one PowerShell window
is needed.

Run from the 07-inference-engines folder:
    python run_pc.py

The 12 GB card holds the 8 GB model plus room for 16 users of about 650
tokens each, so this goes up to 16 users (the Colab T4 had 15 GB and ran 32).
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--server", default=r"C:\Users\brian\tools\llama.cpp\llama-server.exe")
ap.add_argument("--model", default=r"C:\Users\brian\models\Qwen3-4B-F16.gguf")
ap.add_argument("--slots", type=int, default=16)
ap.add_argument("--users", default="1,2,4,8,16")
ap.add_argument("--port", type=int, default=8080)
args = ap.parse_args()

for path in (args.server, args.model):
    if not os.path.exists(path):
        raise SystemExit(f"Not found: {path}")

DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
url = f"http://127.0.0.1:{args.port}"


def health():
    try:
        with DIRECT.open(url + "/health", timeout=2) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


if health() is not None:
    raise SystemExit(f"Something is already using port {args.port}. "
                     "Close other llama-server windows, or run: taskkill /F /IM llama-server.exe")

os.makedirs("logs", exist_ok=True)
log_path = "logs/server_llamacpp_pc.log"
cmd = [args.server, "-m", args.model, "-ngl", "99", "-c", str(1024 * args.slots),
       "-np", str(args.slots), "--cache-ram", "0", "--port", str(args.port)]
print("Starting:", " ".join(cmd))
with open(log_path, "w", encoding="utf-8") as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
try:
    t0 = time.time()
    while health() != 200:
        if proc.poll() is not None:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                print(f.read()[-2500:])
            raise SystemExit("llama-server stopped while loading. The end of its log is above.")
        if time.time() - t0 > 600:
            raise SystemExit("llama-server didn't finish loading in 10 minutes.")
        time.sleep(2)
    print(f"Ready after {time.time() - t0:.0f} s")
    subprocess.run([sys.executable, "engine_bench.py", "--url", url, "--engine", "llama.cpp",
                    "--hardware", "RTX 5070", "--users", args.users], check=True)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Stopped llama-server")
