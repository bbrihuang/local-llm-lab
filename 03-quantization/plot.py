"""Make the Task 3 charts from the CSVs in results/.

Run from the 03-quantization folder:  python plot.py
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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


quality = read("results/quality_kld.csv")
speed = read("results/speed_llama_bench.csv")

# Chart 1: file size vs. KL divergence (F16 is the reference, so it has no KLD point)
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
for imat, color, label in (("no", BLUE, "Plain quantization"),
                           ("yes", ORANGE, "With importance matrix")):
    rows = [r for r in quality if r["imatrix"] == imat and r["quant"] != "F16"]
    rows.sort(key=lambda r: float(r["size_mib"]))
    x = [float(r["size_mib"]) / 1024 for r in rows]
    y = [float(r["mean_kld"]) for r in rows]
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=2, label=label, zorder=3)
    for r, xi, yi in zip(rows, x, y):
        ax.annotate(f'{r["quant"]}  {yi:.3f}', (xi, yi), textcoords="offset points",
                    xytext=(8, 6 if imat == "no" else -14), fontsize=9, color=INK)
ax.set_yscale("log")
style(ax, "Smaller files drift further from the full model",
      "File size (GiB)", "Mean KL divergence vs. F16 (log scale, lower is better)")
ax.legend(frameon=False, labelcolor=INK)
fig.text(0.01, 0.01, "Qwen3.5-4B, WikiText-2 test, 100 chunks of 512 tokens. F16 (8.06 GiB) is the reference.",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("results/size_vs_kld.png")

# Chart 2: file size vs. generation speed
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
rows = sorted(speed, key=lambda r: float(r["file_size_gib"]))
x = [float(r["file_size_gib"]) for r in rows]
y = [float(r["tg128_tok_s"]) for r in rows]
ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=8,
        markeredgecolor="white", markeredgewidth=2, zorder=3)
for r, xi, yi in zip(rows, x, y):
    ax.annotate(f'{r["quant"]}  {yi:.0f} tok/s', (xi, yi), textcoords="offset points",
                xytext=(8, 6), fontsize=9, color=INK)
ax.set_ylim(0, max(y) * 1.15)
style(ax, "Smaller files write text faster",
      "File size (GiB)", "Generation speed (tokens/s, tg128)")
fig.text(0.01, 0.01, "llama-bench on an RTX 5070 (12 GB), all layers on the GPU, 3 runs each. "
         "Importance-matrix files use the same sizes and bit types (speed not re-measured).", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("results/size_vs_speed.png")
print("Saved results/size_vs_kld.png and results/size_vs_speed.png")
