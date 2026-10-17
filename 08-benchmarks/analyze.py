"""Summarize the Task 8 runs: harness scores, MMLU-Pro by subject, and the 30 sample questions.

Run from the 08-benchmarks folder:  ..\\.venv\\Scripts\\python analyze.py
Reads results/harness/ (from run_harness.py), results/model_answers.csv (from ask_model.py) and
results/hand_checks.csv (answers the automatic check couldn't judge, which I looked at by hand).
Writes results/harness_summary.csv, results/mmlu_pro_by_subject.csv,
results/own_questions_summary.csv and results/harness_scores.png.
"""
import csv
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
OFFICIAL_MMLU_PRO = 79.1  # Qwen3.5-4B model card, MMLU-Pro (the card doesn't say thinking or not)


def read_samples(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def stats(rows):
    n = len(rows)
    right = sum(int(r["exact_match"]) for r in rows)
    unanswered = sum(1 for r in rows if r["filtered_resps"][0] == "[invalid]")
    p = right / n
    return dict(n=n, right=right, accuracy=round(100 * p, 1),
                stderr=round(100 * math.sqrt(p * (1 - p) / (n - 1)), 1),
                unanswered=unanswered,
                accuracy_if_answered=round(100 * right / (n - unanswered), 1))


# ---------------------------------------------------------------- harness
runs = []  # (task, label, rows)
for mode in ("instant", "thinking"):
    for task in ("arc_challenge_qwen", "mmlu_pro_qwen"):
        files = sorted(glob.glob(f"results/harness/{mode}/*/samples_{task}_*.jsonl"))
        for i, path in enumerate(files, 1):
            label = "thinking off" if mode == "instant" else (
                "thinking on" if len(files) == 1 else f"thinking on, run {i}")
            runs.append((task, label, read_samples(path)))

with open("results/harness_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["benchmark", "setting", "questions", "right", "accuracy_pct", "stderr_pct",
                "unanswered", "accuracy_pct_if_only_answered"])
    for task, label, rows in runs:
        s = stats(rows)
        w.writerow(["ARC-Challenge" if task.startswith("arc") else "MMLU-Pro", label, s["n"], s["right"],
                    s["accuracy"], s["stderr"], s["unanswered"], s["accuracy_if_answered"]])

by_subject = defaultdict(dict)
for task, label, rows in runs:
    if task != "mmlu_pro_qwen":
        continue
    c = defaultdict(list)
    for r in rows:
        c[r["doc"]["category"]].append(int(r["exact_match"]))
    for subject, v in c.items():
        by_subject[subject][label] = round(100 * sum(v) / len(v))
labels = [l for t, l, _ in runs if t == "mmlu_pro_qwen"]
with open("results/mmlu_pro_by_subject.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["subject", "questions"] + [f"{l} (% right)" for l in labels])
    for subject in sorted(by_subject):
        w.writerow([subject, 15] + [by_subject[subject].get(l, "") for l in labels])

# ---------------------------------------------------------------- the 30 sample questions
checks = {}
with open("results/hand_checks.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        checks[(r["benchmark"], r["id"], r["mode"])] = r["correct"]
tally = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # right, wrong, not yet checked
with open("results/model_answers.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        c = r["correct"] or checks.get((r["benchmark"], r["id"], r["mode"]), "")
        tally[r["benchmark"]][r["mode"]][0 if c == "1" else 1 if c == "0" else 2] += 1
with open("results/own_questions_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["benchmark", "setting", "right", "wrong", "not_yet_checked"])
    for bench, modes in tally.items():
        for mode in ("instant", "thinking"):
            w.writerow([bench, "thinking off" if mode == "instant" else "thinking on", *modes[mode]])

# ---------------------------------------------------------------- chart
rows_to_plot = []  # (group, label, stats)
for task, label, rows in runs:
    rows_to_plot.append(("ARC-Challenge" if task.startswith("arc") else "MMLU-Pro", label, stats(rows)))
order = ["ARC-Challenge", "MMLU-Pro"]
rows_to_plot.sort(key=lambda r: (order.index(r[0]), r[1]))
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
y, ticks, last_group, mmlu_ys = 0, [], None, []
for group, label, s in rows_to_plot:
    if last_group is not None and group != last_group:
        y -= 0.8
    last_group = group
    color = BLUE if label == "thinking off" else ORANGE
    ax.errorbar(s["accuracy"], y, xerr=s["stderr"], fmt="o", ms=9, color=color, mec="white", mew=2,
                ecolor=color, elinewidth=2, capsize=0, zorder=3)
    ax.text(s["accuracy"], y + 0.27, f"{s['accuracy']:.1f}%", ha="center", fontsize=9, color=INK)
    if group == "MMLU-Pro":
        mmlu_ys.append(y)
        ax.plot(s["accuracy_if_answered"], y, "D", ms=8, mfc="white", mec=color, mew=2, zorder=3)
    ticks.append((y, f"{group}\n{label}" if label != "thinking on, run 2" else f"{group}\n{label}"))
    y -= 1
ax.plot([OFFICIAL_MMLU_PRO] * 2, [min(mmlu_ys) - 0.5, max(mmlu_ys) + 0.5], ls=(0, (3, 3)), color=MUTED, lw=1.5)
ax.text(OFFICIAL_MMLU_PRO + 0.4, max(mmlu_ys) + 0.45, f"Qwen's official\nMMLU-Pro: {OFFICIAL_MMLU_PRO}%",
        fontsize=8.5, color=MUTED, va="top")
ax.set_yticks([t[0] for t in ticks])
ax.set_yticklabels([t[1] for t in ticks], fontsize=8.5, color=INK)
ax.set_xlim(64, 100)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.suptitle("Counting unfinished answers as wrong, MMLU-Pro lands 5 to 9 points under Qwen's 79.1",
             x=0.02, ha="left", fontsize=11.5, color=INK, fontweight="bold")
fig.text(0.02, 0.895, "Filled dot: all questions, with one standard error each side. Hollow diamond: only the "
         "questions where it finished an answer\n(not a fair score; it leaves out the ones it ran out of room on).",
         fontsize=8.3, color=MUTED, va="top")
fig.text(0.02, 0.015, "Qwen3.5-4B Q4_K_M on llama-server, RTX 5070. 200 ARC-Challenge and 210 MMLU-Pro questions, "
         "no examples in the prompt.", fontsize=7.8, color=MUTED)
fig.subplots_adjust(left=0.2, right=0.97, top=0.78, bottom=0.1)
fig.savefig("results/harness_scores.png", facecolor="white")
print("Saved results/harness_summary.csv, mmlu_pro_by_subject.csv, own_questions_summary.csv, harness_scores.png")
