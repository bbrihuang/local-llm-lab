"""Summarize the HumanEval runs.

Run from the 09-coding-benchmarks folder:  ..\\.venv\\Scripts\\python analyze.py
Reads results/harness/{instant,thinking}/ (from run_humaneval.py).
Writes results/humaneval_summary.csv, results/humaneval_by_problem.csv and results/humaneval_scores.png.
"""
import csv
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
MODES = [("instant", "thinking off", BLUE), ("thinking", "thinking on", ORANGE)]

runs = {}
for key, _, _ in MODES:
    files = sorted(glob.glob(f"results/harness/{key}/**/samples_humaneval_qwen_*.jsonl", recursive=True))
    if not files:
        continue
    with open(files[-1], encoding="utf-8") as f:  # the newest run
        runs[key] = [json.loads(line) for line in f]
if not runs:
    raise SystemExit("No results found in results/harness/. Run run_humaneval.py first.")

summary = []
for key, label, _ in MODES:
    if key not in runs:
        continue
    rows = runs[key]
    n = len(rows)
    right = sum(int(r["pass_at_1"]) for r in rows)
    no_code = sum(int(r["no_code_block"]) for r in rows)
    p = right / n
    summary.append([label, n, right, round(100 * p, 1), round(100 * math.sqrt(p * (1 - p) / (n - 1)), 1),
                    no_code, round(100 * right / (n - no_code), 1) if n > no_code else ""])
with open("results/humaneval_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["setting", "problems", "passed", "pass_at_1_pct", "std_error_pts", "no_code_block",
                "pass_pct_of_replies_with_code"])
    w.writerows(summary)

by = {k: {r["doc"]["task_id"]: int(r["pass_at_1"]) for r in rows} for k, rows in runs.items()}
ids = sorted(next(iter(by.values())), key=lambda t: int(t.split("/")[1]))
with open("results/humaneval_by_problem.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["task_id"] + [k for k in by])
    for t in ids:
        w.writerow([t] + [by[k].get(t, "") for k in by])
if len(by) == 2:
    a, b = by["instant"], by["thinking"]
    both = sum(1 for t in ids if a[t] and b[t])
    only_a = sum(1 for t in ids if a[t] and not b[t])
    only_b = sum(1 for t in ids if b[t] and not a[t])
    print(f"Passed in both: {both}; only thinking off: {only_a}; only thinking on: {only_b}; "
          f"neither: {len(ids) - both - only_a - only_b}")
for s in summary:
    print(s)

fig, ax = plt.subplots(figsize=(8, 2.8), dpi=150)
for i, (key, label, color) in enumerate(MODES):
    s = next((x for x in summary if x[0] == label), None)
    if not s:
        continue
    y = len(MODES) - i
    ax.errorbar(s[3], y, xerr=s[4], fmt="o", color=color, ms=9, lw=2, capsize=0)
    ax.text(s[3], y + 0.28, f"{s[3]}%", ha="center", color=INK, fontsize=11)
ax.set_yticks([2, 1])
ax.set_yticklabels(["HumanEval\nthinking off", "HumanEval\nthinking on"], color=INK)
ax.set_ylim(0.4, 2.8)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors=MUTED, length=0)
ax.set_title("HumanEval pass@1, Qwen3.5-4B Q4_K_M (one standard error each side)",
             loc="left", fontsize=12, color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig("results/humaneval_scores.png")
print("Saved results/humaneval_scores.png")
