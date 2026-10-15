import csv
import statistics
import requests

URL = "http://127.0.0.1:8080"
TARGETS = [50, 1000, 4000]   # prompt lengths in tokens
REPEATS = 3
GEN_TOKENS = 128

PARAGRAPH = (
    "Computers store information as numbers, and language models turn text into numbers "
    "called tokens. Each token is a small piece of a word, a whole word, or a symbol. "
    "A model reads the tokens it has been given, then predicts which token is most likely "
    "to come next, appends it, and repeats. Rivers carry water from mountains to the sea, "
    "bakers fold butter into dough to make flaky layers, and astronomers measure the light "
    "of distant stars to learn how old they are. "
)


def tokenize(text):
    r = requests.post(f"{URL}/tokenize", json={"content": text})
    r.raise_for_status()
    return r.json()["tokens"]


def detokenize(tokens):
    r = requests.post(f"{URL}/detokenize", json={"tokens": tokens})
    r.raise_for_status()
    return r.json()["content"]


def run(prompt):
    r = requests.post(f"{URL}/completion", json={
        "prompt": prompt,
        "n_predict": GEN_TOKENS,
        "cache_prompt": False,   # force a fresh read of the whole prompt
        "ignore_eos": True,      # always generate the full 128 tokens
        "temperature": 0.7,
    })
    r.raise_for_status()
    return r.json()["timings"]


long_tokens = tokenize(PARAGRAPH * 60)
run(detokenize(long_tokens[:20]))  # warm-up, not recorded

rows = []
for target in TARGETS:
    prompt = detokenize(long_tokens[:target])
    for i in range(REPEATS):
        t = run(prompt)
        rows.append({
            "target_tokens": target,
            "prompt_tokens": t["prompt_n"],
            "time_to_first_token_s": round(t["prompt_ms"] / 1000, 3),
            "prefill_tok_per_s": round(t["prompt_per_second"], 1),
            "decode_tok_per_s": round(t["predicted_per_second"], 1),
            "generated_tokens": t["predicted_n"],
            "repeat": i + 1,
        })
        print(rows[-1])

with open("results/timings_raw.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nAverages over", REPEATS, "runs")
for target in TARGETS:
    r = [x for x in rows if x["target_tokens"] == target]
    print(target, "tokens | prefill",
          round(statistics.mean(x["prefill_tok_per_s"] for x in r), 1), "tok/s | decode",
          round(statistics.mean(x["decode_tok_per_s"] for x in r), 1), "tok/s | first token",
          round(statistics.mean(x["time_to_first_token_s"] for x in r), 3), "s")