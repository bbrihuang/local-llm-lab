"""Make the Task 7 charts and summary table from results/runs.csv.

Run from the 07-inference-engines folder:  ..\\.venv\\Scripts\\python plot.py
Total speed = all users' tokens in a round / wall time of that round; the median of the 3 rounds.
Per-user speed = each request's decode speed once text is flowing; the median request.
Medians, because llama.cpp's second round at some user counts ran at about half speed
(see the README), and an average would blur that into every number.
"""
import csv
import os
import statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SETUPS = [("vLLM", "Colab T4"), ("llama.cpp", "Colab T4"), ("llama.cpp", "RTX 5070")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
NOTE = ("Qwen3-4B, 16-bit weights. Each request: ~400 prompt tokens, 256 generated tokens. "
        "Median of 3 rounds; all users start at the same moment.")

with open("results/runs.csv", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

ttft = defaultdict(list)
decode = defaultdict(list)
rounds = defaultdict(lambda: {"tokens": 0, "wall": 0.0})
for r in rows:
    key = (r["engine"], r["hardware"], int(r["users"]))
    if r["ttft_s"]:
        ttft[key].append(float(r["ttft_s"]))
    if r["decode_tok_s"]:
        decode[key].append(float(r["decode_tok_s"]))
    rnd = rounds[key + (r["round"],)]
    rnd["tokens"] += int(r["tokens"])
    rnd["wall"] = float(r["round_wall_s"])

total = defaultdict(list)
for (engine, hw, users, _), v in rounds.items():
    total[(engine, hw, users)].append(v["tokens"] / v["wall"])

summary = []
for key in sorted(total, key=lambda k: (SETUPS.index(k[:2]) if k[:2] in SETUPS else 99, k[2])):
    summary.append({
        "engine": key[0], "hardware": key[1], "users": key[2],
        "ttft_p50_s": round(statistics.median(ttft[key]), 3),
        "per_user_tok_s": round(statistics.median(decode[key]), 1),
        "total_tok_s": round(statistics.median(total[key]), 1),
        "total_tok_s_best_round": round(max(total[key]), 1),
        "total_tok_s_worst_round": round(min(total[key]), 1),
    })
with open("results/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12)
    ax.set_xlabel("Users sending a request at the same time", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def chart(field, title, ylabel, out, top=None):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ends = []
    for (engine, hw), color, marker in zip(SETUPS, COLORS, MARKERS):
        pts = [(s["users"], s[field]) for s in summary if (s["engine"], s["hardware"]) == (engine, hw)]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=f"{engine} on {hw}")
        ends.append((xs[-1], ys[-1]))
    # Label the last point of each line; nudge apart labels that would overlap.
    top_y = max(y for _, y in ends)
    ends.sort(key=lambda e: (e[0], -e[1]))
    for i, (x, y) in enumerate(ends):
        dy = -3
        if i + 1 < len(ends) and ends[i + 1][0] == x and y - ends[i + 1][1] < 0.05 * top_y:
            dy = 4
        elif i > 0 and ends[i - 1][0] == x and ends[i - 1][1] - y < 0.05 * top_y:
            dy = -10
        ax.annotate(f"{y:,.{1 if y < 100 else 0}f}", (x, y), textcoords="offset points",
                    xytext=(8, dy), fontsize=9, color=INK)
    users = sorted({s["users"] for s in summary})
    ax.set_xticks(users)
    ax.set_xlim(users[0] / 1.3, users[-1] * 1.6)
    ax.set_ylim(0, top)
    style(ax, title, ylabel)
    ax.legend(frameon=False, labelcolor=INK)
    fig.text(0.01, 0.01, NOTE, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out)
    plt.close(fig)


chart("total_tok_s", "More users means more total output, on every setup",
      "Total tokens per second, all users", "results/total_throughput.png")
chart("per_user_tok_s", "But each user's own speed drops as the GPU is shared",
      "Tokens per second for each user", "results/per_user_speed.png")
chart("ttft_p50_s", "And users wait longer before the first word appears",
      "Time to first token, median (seconds)", "results/time_to_first_token.png")

print(f"{'engine':<10} {'hardware':<9} {'users':>5} {'TTFT':>7} {'per-user':>9} {'total':>7}")
for s in summary:
    print(f"{s['engine']:<10} {s['hardware']:<9} {s['users']:>5} {s['ttft_p50_s']:>6.3f}s "
          f"{s['per_user_tok_s']:>9.1f} {s['total_tok_s']:>7.1f}")
print("Saved results/summary.csv and 3 charts in results/")
