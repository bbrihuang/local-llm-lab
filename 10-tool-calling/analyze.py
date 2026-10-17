"""Combine results/tools_*.jsonl into summary tables and a chart.

Run from the 10-tool-calling folder:  ..\\.venv\\Scripts\\python analyze.py
Writes results/tools_summary.csv, results/tools_by_case.csv, results/tools_failures.csv, results/tools_scores.png
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
rows = []
for p in sorted(glob.glob("results/tools_*.jsonl")):
    rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
if not rows:
    raise SystemExit("No results/tools_*.jsonl files found. Run run_tools.py first.")

runs = sorted({(r["label"], r["mode"]) for r in rows})
KINDS = ["single", "multi", "none"]


def pct(xs):
    return 100 * sum(xs) / len(xs) if xs else float("nan")


summary = []
for lab, mode in runs:
    rs = [r for r in rows if r["label"] == lab and r["mode"] == mode]
    d = {"run": lab, "mode": mode, "tries": len(rs),
         "correct_pct": round(pct([r["correct"] for r in rs]), 1),
         "right_tool_pct": round(pct([r["right_tool"] for r in rs]), 1),
         "valid_json_pct": round(pct([r["valid_json"] for r in rs]), 1)}
    for k in KINDS:
        d[f"{k}_correct_pct"] = round(pct([r["correct"] for r in rs if r["kind"] == k]), 1)
    d["median_seconds"] = round(sorted(r["seconds"] for r in rs)[len(rs) // 2], 2)
    summary.append(d)
with open("results/tools_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)

cases = sorted({(r["case"], r["kind"], r["prompt"]) for r in rows})
with open("results/tools_by_case.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["case", "kind", "prompt"] + [f"{l} {m} correct" for l, m in runs])
    for c, k, p in cases:
        line = [c, k, p]
        for lab, mode in runs:
            rs = [r for r in rows if r["label"] == lab and r["mode"] == mode and r["case"] == c]
            line.append(f"{sum(r['correct'] for r in rs)}/{len(rs)}")
        w.writerow(line)

with open("results/tools_failures.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["run", "mode", "case", "prompt", "what_it_did"])
    for r in rows:
        if r["correct"]:
            continue
        calls = "; ".join(f"{c['name']}({json.dumps(c['args'])})" for c in r["calls"])
        w.writerow([r["label"], r["mode"], r["case"], r["prompt"],
                    calls or ("no tool call, replied: " + r["reply_text"][:120].replace("\n", " "))])

for s in summary:
    print(s)

colors = ["#2a6ebb", "#d9822b", "#4a9d6b", "#8c5bb5"]
groups = ["Overall"] + [{"single": "One call", "multi": "Several calls", "none": "No tool needed"}[k] for k in KINDS]
fig, ax = plt.subplots(figsize=(8, 4.2))
width = 0.8 / len(summary)
for i, s in enumerate(summary):
    vals = [s["correct_pct"]] + [s[f"{k}_correct_pct"] for k in KINDS]
    xs = [j + i * width - 0.4 + width / 2 for j in range(len(groups))]
    bars = ax.bar(xs, vals, width * 0.9, color=colors[i % 4], label=f"{s['run']} ({s['mode']})")
    for x, v in zip(xs, vals):
        ax.text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups)
ax.set_ylim(0, 125)
ax.set_ylabel("Correct tries (%)")
ax.set_title("Tool calling, Qwen3.5-4B: 15 requests x repeats")
ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=9)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig("results/tools_scores.png", dpi=150)
print("Saved results/tools_summary.csv, tools_by_case.csv, tools_failures.csv, tools_scores.png")
