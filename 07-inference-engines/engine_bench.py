"""Speed test for any server with an OpenAI-style /v1/completions endpoint.

Works the same against llama-server and vLLM, so both engines get exactly the
same test. For each number of users it sends that many requests at once, with
streaming on, and records:
  - time to first token (how long a user waits before text starts appearing)
  - decode speed per user (tokens per second once text is flowing)
  - total throughput (all users' tokens / wall time of the round)

Every prompt starts with a different request number, so neither engine can
reuse a cached copy of an earlier prompt.

Example:
    python engine_bench.py --url http://localhost:8000 --engine vllm --hardware "Colab T4"
"""
import argparse
import csv
import json
import os
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--engine", required=True, help="label for the CSV, e.g. vllm or llama.cpp")
ap.add_argument("--hardware", required=True, help="label for the CSV, e.g. 'Colab T4'")
ap.add_argument("--model", default="", help="model name to send (vLLM needs it; llama-server ignores it)")
ap.add_argument("--users", default="1,2,4,8,16,32", help="comma-separated concurrency levels")
ap.add_argument("--rounds", type=int, default=3, help="rounds per concurrency level")
ap.add_argument("--prompt-words", type=int, default=380, help="about 500 tokens")
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--out", default="results/runs.csv")
args = ap.parse_args()

DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
FIELDS = ["engine", "hardware", "users", "round", "request", "prompt_tokens", "tokens",
          "ttft_s", "decode_tok_s", "request_s", "round_wall_s"]

WORDS = ("the model reads every word of this passage before it writes anything and the time it "
         "takes to do that is the prefill while the time it then spends writing one token after "
         "another is the decode ").split()
BODY = " ".join(WORDS[i % len(WORDS)] for i in range(args.prompt_words))


def one_request(n, out):
    payload = {
        "prompt": f"Request {n}. {BODY}\nNow continue the story:",
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,          # always generate exactly max_tokens
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.model:
        payload["model"] = args.model
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    chunks = 0
    usage = {}
    with DIRECT.open(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            msg = json.loads(data)
            if msg.get("usage"):
                usage = msg["usage"]
            choices = msg.get("choices") or []
            if choices and choices[0].get("text"):
                chunks += 1
                if first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    tokens = usage.get("completion_tokens") or chunks
    decode = (tokens - 1) / (end - first) if first and end > first and tokens > 1 else None
    out.update(prompt_tokens=usage.get("prompt_tokens", ""), tokens=tokens,
               ttft_s=round(first - t0, 4) if first else "",
               decode_tok_s=round(decode, 2) if decode else "",
               request_s=round(end - t0, 4))


def run_round(users, rnd, counter):
    results = [dict() for _ in range(users)]
    errors = []
    barrier = threading.Barrier(users)

    def worker(i):
        barrier.wait()
        try:
            one_request(counter + i, results[i])
        except Exception as e:  # keep going; report at the end of the round
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(users)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    if errors:
        raise SystemExit(f"{len(errors)} request(s) failed, first error: {errors[0]}")
    rows = []
    for i, r in enumerate(results):
        rows.append(dict(engine=args.engine, hardware=args.hardware, users=users, round=rnd,
                         request=i, round_wall_s=round(wall, 4), **r))
    return rows, wall


# Check the server is up before starting.
try:
    DIRECT.open(args.url.rstrip("/") + "/health", timeout=10).read()
except Exception as e:
    raise SystemExit(f"Can't reach the server at {args.url}: {e}\nIs it running and finished loading?")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
new_file = not os.path.exists(args.out)
counter = 0
print("Warm-up request ...")
run_round(1, 0, 10_000)

print(f"{'users':>5} {'TTFT p50':>9} {'per-user tok/s':>15} {'total tok/s':>12}")
with open(args.out, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
    for users in [int(u) for u in args.users.split(",")]:
        ttfts, decs, thr = [], [], []
        for rnd in range(1, args.rounds + 1):
            rows, wall = run_round(users, rnd, counter)
            counter += users
            w.writerows(rows)
            f.flush()
            ttfts += [r["ttft_s"] for r in rows if r["ttft_s"] != ""]
            decs += [r["decode_tok_s"] for r in rows if r["decode_tok_s"] != ""]
            thr.append(sum(r["tokens"] for r in rows) / wall)
        ttfts.sort()
        print(f"{users:>5} {ttfts[len(ttfts) // 2]:>8.3f}s {sum(decs) / len(decs):>15.1f} "
              f"{sum(thr) / len(thr):>12.1f}")
print(f"Saved {args.out}")
