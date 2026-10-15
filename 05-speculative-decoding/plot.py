"""Make the Task 5 charts and summary tables from results/runs.csv.

Run from the 05-speculative-decoding folder:  python plot.py
Speed = tokens generated in a round / wall time of that round, averaged over runs.
Speedup = speed with the draft model / speed without it, at the same number of users.
"""
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROMPTS = ["code", "facts", "story"]
COLORS = {"code": "#2a78d6", "facts": "#eb6834", "story": "#1baf7a"}
MARKERS = {"code": "o", "facts": "s", "story": "^"}
NAMES = {"code": "Rewriting code", "facts": "Listing facts", "story": "Writing a story"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
USERS_DRAFT = 4  # draft length used for the multi-user runs


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


with open("results/runs.csv", newline="", encoding="utf-8-sig") as f:
    all_rows = list(csv.DictReader(f))
# Main sweep rows are labelled u<users>_d<draft>; anything else is a follow-up run.
rows = [r for r in all_rows if r["label"] == f'u{r["users"]}_d{r["draft_max"]}']
extra_rows = [r for r in all_rows if r not in rows]

# One entry per round: (users, draft, prompt) -> list of round speeds, plus draft counts.
rounds = defaultdict(lambda: {"tokens": 0, "wall": 0.0})
drafts = defaultdict(lambda: [0, 0])
for r in rows:
    key = (int(r["users"]), int(r["draft_max"]), r["prompt"], r["label"], r["run"])
    rounds[key]["tokens"] += int(r["tokens"] or 0)
    rounds[key]["wall"] = float(r["round_wall_s"])
    if r["draft_n"]:
        d = drafts[(int(r["users"]), int(r["draft_max"]), r["prompt"])]
        d[0] += int(r["draft_accepted"])
        d[1] += int(r["draft_n"])

speeds = defaultdict(list)
for (users, draft, prompt, _, _), v in rounds.items():
    speeds[(users, draft, prompt)].append(v["tokens"] / v["wall"])
speed = {k: sum(v) / len(v) for k, v in speeds.items()}


def speedup(users, draft, prompt):
    base = speed.get((users, 0, prompt))
    other = speed.get((users, draft, prompt))
    return other / base if base and other else None


def accept(users, draft, prompt):
    a, n = drafts.get((users, draft, prompt), (0, 0))
    return 100 * a / n if n else None


# Summary tables (also used for the report).
lengths = sorted({d for (u, d, _) in speed if u == 1})
users_list = sorted({u for (u, _, _) in speed})
with open("results/summary_draft_length.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["prompt", "draft_max", "tok_s", "speedup", "accept_pct"])
    for p in PROMPTS:
        for d in lengths:
            if (1, d, p) in speed:
                s, a = speedup(1, d, p), accept(1, d, p)
                w.writerow([p, d, round(speed[(1, d, p)], 1), round(s, 2) if s else "",
                            round(a, 1) if a is not None else ""])
with open("results/summary_users.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["prompt", "users", "baseline_tok_s", f"draft{USERS_DRAFT}_tok_s",
                "speedup", "accept_pct"])
    for p in PROMPTS:
        for u in users_list:
            if (u, 0, p) in speed and (u, USERS_DRAFT, p) in speed:
                a = accept(u, USERS_DRAFT, p)
                w.writerow([p, u, round(speed[(u, 0, p)], 1),
                            round(speed[(u, USERS_DRAFT, p)], 1),
                            round(speedup(u, USERS_DRAFT, p), 2),
                            round(a, 1) if a is not None else ""])


# Follow-up runs (one user), compared with the one-user baseline.
extra = defaultdict(lambda: defaultdict(lambda: {"tokens": 0, "wall": 0.0, "acc": 0, "n": 0}))
for r in extra_rows:
    e = extra[(r["label"], r["prompt"])][r["run"]]
    e["tokens"] += int(r["tokens"] or 0)
    e["wall"] = float(r["round_wall_s"])
    if r["draft_n"]:
        e["acc"] += int(r["draft_accepted"])
        e["n"] += int(r["draft_n"])
with open("results/summary_extra.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["label", "prompt", "tok_s", "speedup", "accept_pct"])
    for (label, p), runs in sorted(extra.items()):
        tps = sum(v["tokens"] / v["wall"] for v in runs.values()) / len(runs)
        acc, n = sum(v["acc"] for v in runs.values()), sum(v["n"] for v in runs.values())
        base = speed.get((1, 0, p))
        w.writerow([label, p, round(tps, 1), round(tps / base, 2) if base else "",
                    round(100 * acc / n, 1) if n else ""])


def line_chart(xs_for, ys_for, xlabel, title, path, note, xticks):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.annotate("no speedup", (xticks[0], 1.0), textcoords="offset points",
                xytext=(-14, 4), fontsize=8, color=MUTED, ha="left")
    for p in PROMPTS:
        xs, ys = xs_for(p), ys_for(p)
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=COLORS[p], linewidth=2, marker=MARKERS[p], markersize=8,
                markeredgecolor="white", markeredgewidth=2, label=NAMES[p], zorder=3)
        ax.annotate(f"{y[-1]:.2f}x", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(8, -3), fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.set_xlim(xticks[0] / 1.3, xticks[-1] * 1.5)
    style(ax, title, xlabel, "Speedup vs. no draft model")
    ax.legend(frameon=False, labelcolor=INK)
    fig.text(0.01, 0.01, note, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


note = ("Qwen3-8B Q4_K_M with Qwen3-0.6B Q8_0 as draft, llama-server b11630, RTX 5070. "
        "Up to 400 tokens per request, temperature 0, 3 runs each.")
draft_lengths = [d for d in lengths if d > 0]
if draft_lengths:
    line_chart(lambda p: draft_lengths,
               lambda p: [speedup(1, d, p) for d in draft_lengths],
               "Maximum draft length (tokens guessed per step)",
               "No draft length beat the big model on its own (one user)",
               "results/speedup_vs_draft_length.png", note, draft_lengths)
if len(users_list) > 1:
    line_chart(lambda p: users_list,
               lambda p: [speedup(u, USERS_DRAFT, p) for u in users_list],
               "Users sending requests at the same time",
               f"More users made the draft model hurt more (draft length {USERS_DRAFT})",
               "results/speedup_vs_users.png", note, users_list)
print("Saved charts and summary tables in results/")
