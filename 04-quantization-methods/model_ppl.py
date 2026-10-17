"""Part 2: what does each quantization method do to the whole model?

Quantizes every linear layer of Qwen3-0.6B's 28 transformer blocks with one method at a time (the embeddings and
the output head stay in 16 bits) and measures perplexity on the WikiText-2 test text. Lower is better; the
unquantized model is the reference.

Run from the 04-quantization-methods folder:
    ..\\.venv\\Scripts\\python model_ppl.py

Results go to results/model_ppl.csv one method at a time. If the run stops, start it again: methods already in
the file are skipped. Delete the file to start over. Takes roughly 15 to 30 minutes on the RTX 5070 (the GPTQ
methods are the slow ones).

Simplification: GPTQ and AWQ here see the inputs of the unquantized model. The real tools quantize block by block
and feed each block the output of the already-quantized blocks before it, which usually helps a little.
"""
import argparse
import csv
import math
import os
import time

import torch

import quant as Q

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--calib", default=r"C:\Users\brian\models\wikitext-2-raw\wiki.train.raw")
ap.add_argument("--text", default=r"C:\Users\brian\models\wikitext-2-raw\wiki.test.raw")
ap.add_argument("--seq", type=int, default=2048, help="tokens per piece of text")
ap.add_argument("--calib-seqs", type=int, default=32, help="pieces of training text for calibration")
ap.add_argument("--chunks", type=int, default=40, help="pieces of test text for perplexity")
ap.add_argument("--methods", default="all", help="comma-separated names from quant.METHODS, or 'all'")
args = ap.parse_args()

FIELDS = ["method", "family", "bits_per_weight", "perplexity", "vs_unquantized", "quantize_seconds", "what"]


@torch.no_grad()
def perplexity(model, chunks):
    dev = next(model.parameters()).device
    nll = 0.0
    for ids in chunks:
        ids = ids[None].to(dev)
        nll += model(ids, labels=ids).loss.float().item()
    return math.exp(nll / len(chunks))


def main(model, tok, out="results"):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "model_ppl.csv")
    done = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            done = {r["method"]: r for r in csv.DictReader(f)}
        if done:
            print(f"{len(done)} methods already in {path}; skipping them (delete the file to start over).")
    methods = list(Q.METHODS) if args.methods == "all" else args.methods.split(",")
    todo = [m for m in methods if m not in done]
    if not todo:
        print("Nothing left to run.")
    else:
        chunks = Q.token_chunks(tok, args.text, args.seq, args.chunks)
        layers = Q.linear_layers(model)
        stats = None
        if any(Q.METHODS[m][0] != "baseline" for m in todo):
            calib = Q.token_chunks(tok, args.calib, args.seq, args.calib_seqs, max_chars=2_000_000)
            t0 = time.time()
            stats, _ = Q.collect_stats(model, calib, list(layers))
            print(f"Calibrated on {len(calib)} x {args.seq} tokens in {time.time() - t0:.0f} s")
        base = float(done["fp"]["perplexity"]) if "fp" in done else None
        for m in todo:
            family, bits, what = Q.METHODS[m]
            t0 = time.time()
            for name, lin in layers.items():
                st = stats[Q.input_key(name)] if stats else None
                Q.set_module(model, name, Q.make_qlinear(m, lin.weight, lin.bias, st))
            tq = time.time() - t0
            ppl = perplexity(model, chunks)
            for name, lin in layers.items():   # put the original layers back
                Q.set_module(model, name, lin)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if m == "fp":
                base = ppl
            row = {"method": m, "family": family, "bits_per_weight": bits, "perplexity": round(ppl, 3),
                   "vs_unquantized": f"{(ppl / base - 1) * 100:+.1f}%" if base else "", "quantize_seconds": round(tq, 1),
                   "what": what}
            done[m] = row
            print(f"{m:22} perplexity {ppl:10.3f}  {row['vs_unquantized']:>9}  (quantized in {tq:.0f} s)")
            with open(path, "w", newline="") as f:   # rewrite after every method so nothing is lost
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(done[k] for k in Q.METHODS if k in done)
    chart(done, out)


def chart(done, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [m for m in Q.METHODS if m in done]
    ppl = [float(done[m]["perplexity"]) for m in names]
    fams = [Q.METHODS[m][0] for m in names]
    fig, ax = plt.subplots(figsize=(9, 7))
    yy = list(range(len(names)))[::-1]
    ax.barh(yy, ppl, color=[Q.FAMILY_COLORS[f] for f in fams], height=0.7)
    for y, v in zip(yy, ppl):
        ax.text(v * 1.05, y, f"{v:,.1f}", va="center", fontsize=8, color="#333333")
    if "fp" in done:
        ax.axvline(float(done["fp"]["perplexity"]), color="#555555", lw=1, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("WikiText-2 perplexity (log scale, lower is better; dashed line = not quantized)")
    ax.set_title("Qwen3-0.6B with every linear layer quantized")
    for f, c in Q.FAMILY_COLORS.items():
        if f in fams:
            ax.bar(0, 0, color=c, label=f)
    ax.legend(frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "model_ppl.png"), dpi=150)
    plt.close(fig)
    print("Saved", os.path.join(out, "model_ppl.png"))


if __name__ == "__main__":
    model, tok = Q.load_model(args.model)
    main(model, tok)
