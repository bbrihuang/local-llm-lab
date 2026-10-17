"""Part 1: how much error does each quantization method add to a single layer?

Runs Qwen3-0.6B over WikiText-2 and grabs the real inputs of 12 linear layers (4 kinds of layer at 3 depths).
For every method in quant.py it quantizes the layer and measures the relative output error
    ||x W^T - (quantized version)||  /  ||x W^T||
on text the calibration never saw. It also records how big the outlier features in each layer's input are,
because outliers are the reason most of these methods exist.

Run from the 04-quantization-methods folder:
    ..\\.venv\\Scripts\\python layer_error.py

Writes results/layer_error.csv, results/outliers.csv and three charts. Takes a few minutes on the RTX 5070.
"""
import argparse
import csv
import os
import statistics
import time

import torch

import quant as Q

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--calib", default=r"C:\Users\brian\models\wikitext-2-raw\wiki.train.raw")
ap.add_argument("--text", default=r"C:\Users\brian\models\wikitext-2-raw\wiki.test.raw")
ap.add_argument("--layers", default="2,14,26", help="which transformer blocks to test")
ap.add_argument("--modules", default="q_proj,o_proj,up_proj,down_proj")
ap.add_argument("--seq", type=int, default=512, help="tokens per piece of text")
ap.add_argument("--calib-seqs", type=int, default=64, help="pieces used for calibration (GPTQ, AWQ, SmoothQuant)")
ap.add_argument("--eval-seqs", type=int, default=8, help="pieces used to measure the error")
args = ap.parse_args()

NICE = {"q_proj": "attention input", "o_proj": "o_proj input", "gate_proj": "MLP input", "down_proj": "down_proj input"}


def full_name(layer, mod):
    return f"model.layers.{layer}.{'self_attn' if mod in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp'}.{mod}"


def main(model, tok, out="results"):
    os.makedirs(out, exist_ok=True)
    layers = Q.linear_layers(model)
    picked = [full_name(int(l), m) for l in args.layers.split(",") for m in args.modules.split(",")]
    calib = Q.token_chunks(tok, args.calib, args.seq, args.calib_seqs, max_chars=2_000_000)
    evals = Q.token_chunks(tok, args.text, args.seq, args.eval_seqs)
    print(f"Calibration: {len(calib)} x {args.seq} tokens. Test: {len(evals)} x {args.seq} tokens.")

    t0 = time.time()
    stats, _ = Q.collect_stats(model, calib, picked)
    # second pass on the test text: keep the picked layers' inputs, and every layer's per-feature maximum
    outl, X = Q.collect_stats(model, evals, list(layers), want_H=False, capture=picked)
    print(f"Collected activations in {time.time() - t0:.0f} s")

    rows = []
    for name in picked:
        lin = layers[name]
        x = X[name]
        y = x @ lin.weight.float().T
        st = stats[Q.input_key(name)]
        line = []
        for method, (family, bits, _) in Q.METHODS.items():
            if method == "fp":
                continue
            with torch.no_grad():
                ql = Q.make_qlinear(method, lin.weight, None, st)
                err = ((ql(x) - y).norm() / y.norm()).item()
            rows.append({"layer": name.split(".")[2], "module": name.split(".")[-1], "method": method,
                         "family": family, "bits_per_weight": bits, "relative_error": round(err, 5)})
            line.append(f"{method}={err:.3f}")
            del ql
        print(name, " ".join(line[:4]), "...")
    with open(os.path.join(out, "layer_error.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    orow = []
    for key, st in outl.items():
        if not key.endswith(("q_proj", "o_proj", "gate_proj", "down_proj")):
            continue
        xmax = st["xmax"]
        orow.append({"layer": int(key.split(".")[2]), "input": NICE[key.split(".")[-1]],
                     "largest": round(xmax.max().item(), 2), "median_feature_max": round(xmax.median().item(), 3),
                     "ratio": round((xmax.max() / xmax.median()).item(), 1), "top_feature": int(xmax.argmax()),
                     "features_over_6": int((xmax > 6).sum())})
    orow.sort(key=lambda r: (r["input"], r["layer"]))
    with open(os.path.join(out, "outliers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(orow[0]))
        w.writeheader()
        w.writerows(orow)

    # the feature maxima of one layer, for the outlier picture
    mid = args.layers.split(",")[len(args.layers.split(",")) // 2]
    profile = {NICE[m]: outl[full_name(int(mid), m)]["xmax"].cpu() for m in ("q_proj", "down_proj")}
    charts(rows, orow, profile, mid, out)
    summary(rows)


def summary(rows):
    print("\nMedian relative output error over the tested layers:")
    for method, (family, bits, what) in Q.METHODS.items():
        e = [r["relative_error"] for r in rows if r["method"] == method]
        if e:
            print(f"  {method:22} {statistics.median(e):8.4f}   {what}")


def charts(rows, orow, profile, mid, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. median error per method
    methods = [m for m in Q.METHODS if m != "fp"]
    med = [statistics.median(r["relative_error"] for r in rows if r["method"] == m) for m in methods]
    fams = [Q.METHODS[m][0] for m in methods]
    fig, ax = plt.subplots(figsize=(9, 7))
    yy = range(len(methods))[::-1]
    ax.barh(list(yy), med, color=[Q.FAMILY_COLORS[f] for f in fams], height=0.7)
    for y, v in zip(yy, med):
        ax.text(v * 1.05, y, f"{v:.3f}", va="center", fontsize=8, color="#333333")
    ax.set_yticks(list(yy))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Relative output error of one layer (median of the tested layers, log scale, lower is better)")
    ax.set_title("Qwen3-0.6B: error added by each quantization method")
    for f, c in Q.FAMILY_COLORS.items():
        if f in fams:
            ax.bar(0, 0, color=c, label=f)
    ax.legend(frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "layer_error.png"), dpi=150)
    plt.close(fig)

    # 2. outlier size by depth
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#2a6ebb", "#d9822b", "#2a9d8f", "#8e5ea2"]
    for c, name in zip(colors, ["attention input", "o_proj input", "MLP input", "down_proj input"]):
        pts = [r for r in orow if r["input"] == name]
        ax.plot([r["layer"] for r in pts], [r["ratio"] for r in pts], color=c, lw=2, marker="o", ms=4, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Largest feature / median feature")
    ax.set_title("How much bigger the biggest input feature is than a typical one")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "outliers_by_layer.png"), dpi=150)
    plt.close(fig)

    # 3. every feature's largest value in one layer: the outliers stick out
    fig, axes = plt.subplots(len(profile), 1, figsize=(9, 2.6 * len(profile)))
    for ax, (name, v) in zip(axes, profile.items()):
        ax.bar(range(len(v)), v.numpy(), width=1.0, color="#2a6ebb")
        ax.set_title(f"Layer {mid}, {name}: largest |value| of each feature on the test text", fontsize=10)
        ax.set_ylabel("max |x|")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("Feature (hidden dimension)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "outlier_features.png"), dpi=150)
    plt.close(fig)
    print("Saved the charts in", out)


if __name__ == "__main__":
    model, tok = Q.load_model(args.model)
    main(model, tok)
