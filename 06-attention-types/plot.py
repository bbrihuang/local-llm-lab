"""Make the Task 6 charts from results/k90.npz.

Run from the 06-attention-types folder:  ..\\.venv\\Scripts\\python plot.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"

d = np.load("results/k90.npz")
k90, sink, local = d["k90"], d["sink"], d["local"]
T, L, W = int(d["tokens"]), int(d["last"]), int(d["local_window"])
n_layers, n_heads = k90.shape[:2]
layers = np.arange(n_layers)
note = (f"Qwen3-0.6B (bf16, transformers), WikiText-2 test text. {k90.shape[2] // L} documents of "
        f"{T:,} tokens; measured at the last {L:,} positions of each.")


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


# 1. Tokens needed for 90% of attention, per layer.
med = np.median(k90.reshape(n_layers, -1), axis=1)
p25 = np.percentile(k90.reshape(n_layers, -1), 25, axis=1)
p75 = np.percentile(k90.reshape(n_layers, -1), 75, axis=1)
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.fill_between(layers, p25, p75, color=BLUE, alpha=0.15, linewidth=0,
                label="Middle half of heads and positions")
ax.plot(layers, med, color=BLUE, linewidth=2, marker="o", markersize=6,
        markeredgecolor="white", markeredgewidth=1.5, label="Median")
ax.axhline(T - L // 2, color=MUTED, linewidth=1, linestyle="--")
ax.annotate(f"all ~{T - L // 2:,} past tokens", (0, T - L // 2), textcoords="offset points",
            xytext=(0, 4), fontsize=8, color=MUTED)
ax.set_yscale("log")
ax.set_ylim(1, T * 5)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
style(ax, "A typical head gets 90% of its attention from about 34 of 3,500+ tokens",
      "Layer", "Past tokens holding 90% of the attention")
ax.legend(frameon=False, labelcolor=INK, loc="upper right", ncol=2)
fig.text(0.01, 0.01, note, fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("results/k90_by_layer.png")
plt.close(fig)

# 2. Every head: one cell per (layer, head).
heads = np.median(k90, axis=2)
fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
im = ax.imshow(heads, cmap="Blues", norm=LogNorm(vmin=1, vmax=T), aspect="auto")
ax.set_xticks(range(n_heads))
ax.set_yticks(range(0, n_layers, 2))
style(ax, "Heads differ a lot: some use 1 token, a few use over 1,000",
      "Query head (pairs 0-1, 2-3, ... share one key/value head)", "Layer")
for side in ("left", "bottom"):
    ax.spines[side].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Median past tokens for 90% of attention", color=MUTED)
cb.ax.tick_params(colors=MUTED)
cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
cb.outline.set_visible(False)
fig.text(0.01, 0.01, note, fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("results/k90_heatmap.png")
plt.close(fig)

# 3. Where the attention goes: the very first token vs. the most recent ones.
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
s = 100 * sink.reshape(n_layers, -1).mean(axis=1)
r = 100 * local.reshape(n_layers, -1).mean(axis=1)
ax.plot(layers, r, color=BLUE, linewidth=2, marker="o", markersize=6,
        markeredgecolor="white", markeredgewidth=1.5, label=f"Last {W} tokens")
ax.plot(layers, s, color=ORANGE, linewidth=2, marker="s", markersize=6,
        markeredgecolor="white", markeredgewidth=1.5, label="The first token only")
ax.set_ylim(0, 100)
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
style(ax, "From layer 3 on, the first token soaks up a large share of the attention",
      "Layer", "Share of attention (%)")
ax.legend(frameon=False, labelcolor=INK)
fig.text(0.01, 0.01, note, fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("results/sink_and_local.png")
plt.close(fig)

print("Saved results/k90_by_layer.png, results/k90_heatmap.png and results/sink_and_local.png")
