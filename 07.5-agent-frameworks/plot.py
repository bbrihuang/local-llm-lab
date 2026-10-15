"""Make the Task 7.5 charts and summary table from results/runs.csv.

Run from the 07.5-agent-frameworks folder:  ..\\.venv\\Scripts\\python plot.py

results/audit.csv holds the few answers I checked by hand and re-scored, with the reason.
The automatic check only looks for the right number or word anywhere in the answer, so an
answer that lists every option can pass by accident.
"""
import ast
import csv
import os
import statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FRAMEWORKS = ["plain", "langgraph", "openai_agents", "pydantic_ai", "smolagents"]
NAMES = {"plain": "No framework", "langgraph": "LangGraph", "openai_agents": "OpenAI Agents SDK",
         "pydantic_ai": "PydanticAI", "smolagents": "smolagents"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
NOTE = ("Qwen3-8B Q4_K_M on llama-server (RTX 5070), thinking off, temperature 0.7, replies capped at "
        "1,024 tokens, 8,192-token context. 10 questions x 3 repeats per framework.")

with open("results/runs.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
with open("results/audit.csv", newline="", encoding="utf-8") as f:
    audit = {(a["framework"], a["task"], a["repeat"]): int(a["correct"]) for a in csv.DictReader(f)}
for r in rows:
    r["correct_audited"] = audit.get((r["framework"], r["task"], r["repeat"]), int(r["correct"]))


def code_lines(path):
    """Lines of code in an agent file, not counting blank lines, comments or the docstring."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    doc = set()
    if ast.get_docstring(tree):
        doc = set(range(tree.body[0].lineno, tree.body[0].end_lineno + 1))
    return sum(1 for i, line in enumerate(src.splitlines(), 1)
               if line.strip() and not line.strip().startswith("#") and i not in doc)


summary = []
for fw in FRAMEWORKS:
    r = [x for x in rows if x["framework"] == fw]
    if not r:
        continue
    summary.append({
        "framework": fw,
        "lines_of_code": code_lines(f"agent_{fw}.py"),
        "correct": sum(x["correct_audited"] for x in r),
        "correct_automatic": sum(int(x["correct"]) for x in r),
        "runs": len(r),
        "errors": sum(1 for x in r if x["error"]),
        "median_seconds": round(statistics.median(float(x["seconds"]) for x in r), 2),
        "llm_calls": round(statistics.mean(int(x["llm_calls"]) for x in r), 2),
        "tool_calls": round(statistics.mean(int(x["tool_calls"]) for x in r), 2),
        "prompt_tokens": round(statistics.mean(int(x["prompt_tokens"]) for x in r)),
        "completion_tokens": round(statistics.mean(int(x["completion_tokens"]) for x in r)),
        # t03 needs no tools, so its single request shows each framework's fixed overhead
        "no_tools_tokens_per_request": round(statistics.mean(int(x["prompt_tokens"]) / int(x["llm_calls"])
                                                      for x in r if x["task"] == "t03")),
    })
with open("results/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)

fws = [s["framework"] for s in summary]
tasks = sorted({x["task"] for x in rows})
SHORT = {"calculator": "calculator", "no tools needed": "no tools", "list files": "list files",
         "read a file": "read file", "read + calculate": "read + math",
         "read + filter + calculate": "filter + math"}
kinds = {x["task"]: SHORT.get(x["kind"], x["kind"]) for x in rows}
score = defaultdict(int)
for x in rows:
    score[(x["framework"], x["task"])] += x["correct_audited"]


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


# 1. Which questions each framework got right.
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
grid = [[score[(fw, t)] for t in tasks] for fw in fws]
ax.imshow(grid, cmap="Blues", vmin=-0.6, vmax=3, aspect="auto")
for i, fw in enumerate(fws):
    for j, t in enumerate(tasks):
        v = grid[i][j]
        ax.text(j, i, f"{v}/3", ha="center", va="center", fontsize=9,
                color="white" if v >= 2 else INK)
ax.set_xticks(range(len(tasks)))
ax.set_xticklabels([f"{t}\n{kinds[t]}" for t in tasks], fontsize=7, color=MUTED)
ax.set_yticks(range(len(fws)))
ax.set_yticklabels([NAMES[f] for f in fws], color=INK)
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)
ax.set_title("Four frameworks miss the same two questions; smolagents misses most", loc="left",
             fontsize=13, color=INK, pad=12)
fig.text(0.01, 0.01, NOTE, fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("results/correct_by_question.png")
plt.close(fig)

# 2. What each framework costs: tokens sent and time per question (log scale: smolagents is far off).
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
labels = [NAMES[f] for f in fws][::-1]
for ax, field, title, fmt in (
        (axes[0], "prompt_tokens", "Prompt tokens sent per question (average)", "{:,.0f}"),
        (axes[1], "median_seconds", "Seconds per question (median)", "{:,.1f}")):
    vals = [s[field] for s in summary][::-1]
    colors = [ORANGE if f == "smolagents" else BLUE for f in fws][::-1]
    ax.barh(labels, vals, color=colors, height=0.6)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for y, v in enumerate(vals):
        ax.text(v * 1.08, y, fmt.format(v), va="center", fontsize=9, color=INK)
    ax.set_xlim(min(vals) / 2, max(vals) * 4)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    tidy(ax)
    ax.tick_params(axis="y", colors=INK, length=0)
axes[1].set_yticklabels([])
fig.suptitle("smolagents sends about 20x the tokens and takes about 50x longer", x=0.01, ha="left",
             fontsize=13, color=INK)
fig.text(0.01, 0.01, NOTE + " Log scale.", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig("results/cost_per_question.png")
plt.close(fig)

print(f"{'framework':<14} {'code':>5} {'correct':>8} {'errors':>7} {'median s':>9} {'calls':>6} "
      f"{'tokens in':>10} {'no-tools req':>12}")
for s in summary:
    print(f"{s['framework']:<14} {s['lines_of_code']:>5} {s['correct']:>4}/{s['runs']:<3} {s['errors']:>7} "
          f"{s['median_seconds']:>9.1f} {s['llm_calls']:>6.1f} {s['prompt_tokens']:>10,} "
          f"{s['no_tools_tokens_per_request']:>12,}")
print("Saved results/summary.csv, results/correct_by_question.png and results/cost_per_question.png")
