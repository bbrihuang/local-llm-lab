"""Speculative decoding benchmark: starts llama-server, sends prompts, records speed.

Usage (one PowerShell window, from any folder):
    python spec_bench.py --sweep             # everything: about 15 minutes
    python spec_bench.py                     # one run: big model alone, 1 user
    python spec_bench.py --draft 4           # one run: with the draft model, 4 drafted tokens
    python spec_bench.py --draft 4 --users 4 # one run: 4 users at the same time
    python spec_bench.py --extra             # follow-ups: confidence cut-off, n-gram lookup

Results are appended to results/runs.csv (one row per request) and each run's
server output is saved to results/server_<label>.log. Standard library only.
Flags match llama.cpp build b11630 (speculative flags were renamed in 2026).
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SERVER = r"C:\Users\brian\tools\llama.cpp\llama-server.exe"
MAIN_MODEL = r"C:\Users\brian\models\Qwen3-8B-Q4_K_M.gguf"
DRAFT_MODEL = r"C:\Users\brian\models\Qwen3-0.6B-Q8_0.gguf"
PORT = 8080
BASE = f"http://127.0.0.1:{PORT}"
CTX_PER_USER = 1536  # prompt (<250 tokens) + 400 generated + draft room
HEADER = ["label", "users", "draft_max", "prompt", "run", "request", "tokens",
          "gen_tok_s", "draft_n", "draft_accepted", "accept_pct", "truncated",
          "wall_s", "round_wall_s"]

# The full sweep: every draft length for 1 user, then baseline vs. draft 4 under load.
SWEEP = [(1, 0), (1, 2), (1, 4), (1, 8), (1, 16),
         (2, 0), (2, 4), (4, 0), (4, 4), (8, 0), (8, 4)]

# Follow-up runs, one user each: (label, draft length, extra server flags).
# pmin07: the small model stops guessing once it is less than 70% sure.
# ngram: no small model at all; guesses are copied from earlier text (prompt lookup).
EXTRAS = [("u1_d4_pmin07", 4, ["--spec-draft-p-min", "0.7"]),
          ("u1_ngram", 0, ["--spec-type", "ngram-simple", "--spec-draft-n-max", "64"])]

# Talk to the local server directly, never through a Windows system proxy.
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))

CODE = '''def process(data):
    result = []
    for i in range(len(data)):
        if data[i] != None:
            if data[i] > 0:
                result.append(data[i] * 2)
            else:
                result.append(0)
    total = 0
    for x in result:
        total = total + x
    avg = total / len(result)
    return result, total, avg'''

PROMPTS = {
    "code": "Rewrite this Python function so it uses a list comprehension, "
            "has type hints and a docstring, and handles an empty list. "
            "Keep the same behaviour. Reply with only the code.\n\n" + CODE,
    "story": "Write an original short story about a lighthouse keeper who "
             "finds a strange machine washed up on the beach.",
    "facts": "List 25 facts about the planets of the solar system, "
             "one per line, numbered.",
}


def qwen_chat(user_text):
    # Qwen3 chat format with thinking turned off (empty think block).
    return ("<|im_start|>user\n" + user_text + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def health():
    """Return the HTTP status of /health, or None if nothing is listening."""
    try:
        with DIRECT.open(BASE + "/health", timeout=2) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code  # 503 while the model is still loading
    except (urllib.error.URLError, OSError):
        return None


def tail(path, n=25):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(no server log)"


def prepare_csv(path):
    """Make sure results/runs.csv is writable and has the current columns."""
    try:
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8-sig") as f:
                same = next(csv.reader(f), None) == HEADER
            if not same:
                old = os.path.join("results", time.strftime("runs_old_%Y%m%d_%H%M%S.csv"))
                os.replace(path, old)
                print(f"runs.csv had different columns, so it was moved to {old}")
        open(path, "a").close()
    except PermissionError:
        sys.exit(r"results\runs.csv is open in another program (Excel?). Close it and run again.")


def labels_done(path, runs):
    """Labels that already have every row they should, so a sweep can resume."""
    counts = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            counts[row["label"]] = counts.get(row["label"], 0) + 1
    return {lab for lab, n in counts.items()
            if n >= len(PROMPTS) * runs * int(re.match(r"u(\d+)", lab).group(1))}


def stop_server(proc, log):
    if proc.poll() is None:
        proc.terminate()  # a hard kill on Windows; frees the GPU memory
        proc.wait(timeout=15)
    log.close()


def wait_until_ready(proc, log_path, timeout):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            sys.exit("The server stopped while loading. Last lines of its log "
                     f"({log_path}):\n\n" + tail(log_path))
        if health() == 200:
            print(f"  server ready after {time.time() - start:.0f} s")
            return
        time.sleep(1)
    sys.exit(f"The server did not become ready within {timeout} s. "
             f"Last lines of its log ({log_path}):\n\n" + tail(log_path))


def run_one(prompt, n_predict):
    body = json.dumps({
        "prompt": qwen_chat(prompt),
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(BASE + "/completion", body,
                                 {"Content-Type": "application/json"})
    start = time.time()
    with DIRECT.open(req, timeout=900) as r:
        out = json.load(r)
    t = out.get("timings", {})
    return {
        "tokens": t.get("predicted_n"),
        "gen_tok_s": round(t.get("predicted_per_second", 0), 2),
        "draft_n": t.get("draft_n", ""),
        "draft_accepted": t.get("draft_n_accepted", ""),
        "truncated": out.get("truncated", ""),
        "wall_s": round(time.time() - start, 2),
    }


def run_round(prompt, n_predict, users):
    """Send the same prompt from `users` threads at once; return results and wall time."""
    results = [None] * users
    errors = []
    gate = threading.Barrier(users + 1)

    def worker(i):
        gate.wait()
        try:
            results[i] = run_one(prompt, n_predict)
        except Exception as e:  # reported below, after all threads finish
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(users)]
    for th in threads:
        th.start()
    gate.wait()
    start = time.time()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return results, round(time.time() - start, 2)


def bench(users, draft, args, csv_path, label=None, extra=()):
    label = label or f"u{users}_d{draft}"
    log_path = os.path.join("results", f"server_{label}.log")
    if health() is not None:
        sys.exit(f"Something is already running on port {PORT}. Run this, then try again:\n"
                 "    taskkill /F /IM llama-server.exe")

    cmd = [args.server, "-m", args.model, "-ngl", "99", "-c", str(CTX_PER_USER * users),
           "-np", str(users), "--cache-ram", "0", "--port", str(PORT)]
    if draft:
        cmd += ["-md", args.draft_model, "-ngld", "99",
                "--spec-type", "draft-simple", "--spec-draft-n-max", str(draft)]
    cmd += list(extra)
    print(f"\n=== {users} user(s), " + (f"draft length {draft}" if draft else "no draft model")
          + (f", {' '.join(extra)}" if extra else ""))

    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_until_ready(proc, log_path, args.load_timeout)
        if "no usable GPU found" in tail(log_path, 400):
            sys.exit(f"llama-server is running on the CPU, not the GPU. See {log_path}.")
        run_round("Say hi.", 8, users)  # warm-up, not recorded
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if os.path.getsize(csv_path) == 0:
                w.writerow(HEADER)
            for name, prompt in PROMPTS.items():
                for i in range(1, args.runs + 1):
                    results, round_wall = run_round(prompt, args.tokens, users)
                    if draft and all(r["draft_n"] == "" for r in results):
                        sys.exit("The draft model loaded but no tokens were drafted, so "
                                 f"speculative decoding is not on. See {log_path}.")
                    for j, r in enumerate(results, 1):
                        pct = ""
                        if r["draft_n"]:
                            pct = round(100 * r["draft_accepted"] / r["draft_n"], 1)
                        w.writerow([label, users, draft, name, i, j, r["tokens"],
                                    r["gen_tok_s"], r["draft_n"], r["draft_accepted"],
                                    pct, r["truncated"], r["wall_s"], round_wall])
                    f.flush()
                    total = sum(r["tokens"] or 0 for r in results)
                    accepted = sum(r["draft_accepted"] or 0 for r in results)
                    drafted = sum(r["draft_n"] or 0 for r in results)
                    print(f"  {name:6} run {i}: {total / round_wall:7.1f} tok/s"
                          + (f" total for {users} users" if users > 1 else "")
                          + (f", accepted {100 * accepted / drafted:.0f}% of drafts"
                             if drafted else ""))
    finally:
        stop_server(proc, log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="run every configuration")
    ap.add_argument("--extra", action="store_true", help="run the follow-up configurations")
    ap.add_argument("--draft", type=int, default=0,
                    help="max drafted tokens per step; 0 = no draft model")
    ap.add_argument("--users", type=int, default=1, help="requests sent at the same time")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--server", default=SERVER)
    ap.add_argument("--model", default=MAIN_MODEL)
    ap.add_argument("--draft-model", default=DRAFT_MODEL)
    ap.add_argument("--load-timeout", type=int, default=300)
    args = ap.parse_args()

    # Keep results next to this script, wherever it is started from.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "runs.csv")
    prepare_csv(csv_path)
    for path in (args.server, args.model, args.draft_model):
        if not os.path.exists(path):
            sys.exit(f"File not found: {path}")

    if args.extra:
        for n, (label, draft, flags) in enumerate(EXTRAS, 1):
            print(f"\n[{n}/{len(EXTRAS)}]", end="")
            bench(1, draft, args, csv_path, label, flags)
    elif not args.sweep:
        bench(args.users, args.draft, args, csv_path)
    else:
        done = labels_done(csv_path, args.runs)
        todo = [(u, d) for u, d in SWEEP if f"u{u}_d{d}" not in done]
        if len(todo) < len(SWEEP):
            print(f"Skipping {len(SWEEP) - len(todo)} configuration(s) already in runs.csv.")
        for n, (u, d) in enumerate(todo, 1):
            print(f"\n[{n}/{len(todo)}]", end="")
            bench(u, d, args, csv_path)
    print("\nDone. Results are in results\\runs.csv")


if __name__ == "__main__":
    main()
