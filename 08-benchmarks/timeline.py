"""Timeline chart: when each benchmark came out, and when top models maxed it out.

Run from the 08-benchmarks folder:  ..\\.venv\\Scripts\\python timeline.py
Reads data/benchmarks.csv (sources for every date and score are in data/benchmarks_sources.md)
and writes results/benchmark_timeline.png.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
TODAY = "2026-10"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9890"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
STYLE = {"yes": (BLUE, "Maxed out"), "near": (ORANGE, "Close to maxed out"), "no": (GRAY, "Not maxed out yet")}


def year(ym):
    y, m = ym.split("-")
    return int(y) + (int(m) - 1) / 12


rows = sorted(csv.DictReader(open("data/benchmarks.csv", encoding="utf-8")), key=lambda r: r["released"])
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
for i, r in enumerate(rows):
    y = len(rows) - 1 - i
    color, _ = STYLE[r["saturated"]]
    start = year(r["released"])
    end = year(r["saturated_date"]) if r["saturated_date"] else year(TODAY)
    ax.plot([start, end], [y, y], color=color, lw=2, solid_capstyle="round",
            linestyle="-" if r["saturated_date"] else (0, (2, 2)))
    ax.plot(start, y, "o", ms=8, color=color, mec="white", mew=2)
    if r["saturated_date"]:
        ax.plot(end, y, "D", ms=8, color=color, mec="white", mew=2)
        label = f"{end - start:.1f} years" if end > start else "high scores from the start"
    else:
        label = f"best so far {r['best_score']}"
    ax.text(max(end, start) + 0.15, y, label, va="center", fontsize=8.5, color=MUTED)
    name = r["benchmark"].replace(" (HLE)", "")
    ax.text(start - 0.15, y, name, va="center", ha="right", fontsize=9.5, color=INK)

ax.set_yticks([])
ax.set_xlim(2016.2, 2028.6)
ax.set_xticks(range(2018, 2027))
ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
handles = [plt.Line2D([], [], color=c, lw=2, marker="D", ms=6, mec="white", label=l) for c, l in STYLE.values()]
handles.insert(0, plt.Line2D([], [], color=MUTED, lw=0, marker="o", ms=6, mec="white", label="Released"))
ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5, ncol=4, bbox_to_anchor=(0.0, -0.12))
fig.suptitle("Benchmarks get maxed out faster: 2.6 to 5 years for the 2018-2020 ones, 1 to 2 years for 2023-2024",
             x=0.02, ha="left", fontsize=11.5, color=INK, fontweight="bold")
fig.text(0.02, 0.905, "Released (circle) to the month top models first came close to the ceiling (diamond). "
         "Dashed: not there yet (October 2026).", fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.12)
os.makedirs("results", exist_ok=True)
fig.savefig("results/benchmark_timeline.png", facecolor="white")
print("Saved results/benchmark_timeline.png")
