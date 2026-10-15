import csv
import statistics
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/timings_raw.csv")))
lengths = sorted({int(r["target_tokens"]) for r in rows})


def avg(col, n):
    return statistics.mean(float(r[col]) for r in rows if int(r["target_tokens"]) == n)


prefill = [avg("prefill_tok_per_s", n) for n in lengths]
decode = [avg("decode_tok_per_s", n) for n in lengths]
ttft = [avg("time_to_first_token_s", n) for n in lengths]

# Chart 1: reading speed vs writing speed
x = list(range(len(lengths)))
w = 0.38
fig, ax = plt.subplots(figsize=(7.5, 4.8))
b1 = ax.bar([i - w / 2 for i in x], prefill, w, label="Prefill (reading the prompt)")
b2 = ax.bar([i + w / 2 for i in x], decode, w, label="Decode (writing the answer)")
ax.set_yscale("log")
ax.set_ylim(10, 100000)
ax.set_xticks(x)
ax.set_xticklabels([f"{n} tokens" for n in lengths])
ax.set_ylabel("Tokens per second (log scale)")
ax.set_title("Qwen3-1.7B Q8_0 on RTX 5070: prefill vs decode")
ax.bar_label(b1, fmt="%.0f")
ax.bar_label(b2, fmt="%.0f")
ax.legend()
fig.tight_layout()
fig.savefig("results/prefill_vs_decode.png", dpi=150)

# Chart 2: time to first token
fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
ax2.plot(lengths, ttft, marker="o")
for n, t in zip(lengths, ttft):
    ax2.annotate(f"{t:.3f} s", (n, t), textcoords="offset points", xytext=(5, 8))
ax2.set_xlabel("Prompt length (tokens)")
ax2.set_ylabel("Time to first token (seconds)")
ax2.set_title("Waiting time grows with prompt length")
fig2.tight_layout()
fig2.savefig("results/time_to_first_token.png", dpi=150)
print("saved charts")