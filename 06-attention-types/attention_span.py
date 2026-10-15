"""How many earlier tokens does each attention head really use?

Runs Qwen3-0.6B over long stretches of WikiText and, for every layer and head,
counts how many past tokens it takes to cover 90% of a token's attention.
If a head needs only a few dozen tokens out of 4,000, a sliding window or a
sparse pattern could replace full attention there with little loss.

Run from the 06-attention-types folder:
    ..\\.venv\\Scripts\\python attention_span.py

Writes results/per_head.csv, results/per_layer.csv and results/k90.npz.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--text", default=r"C:\Users\brian\models\wikitext-2-raw\wiki.test.raw")
ap.add_argument("--tokens", type=int, default=4096, help="length of each document")
ap.add_argument("--docs", type=int, default=3, help="how many separate 4K stretches to run")
ap.add_argument("--last", type=int, default=1024,
                help="measure only the last N positions, so every query has 3,000+ tokens behind it")
ap.add_argument("--local", type=int, default=128, help="window size for the 'local share' column")
args = ap.parse_args()

if not torch.cuda.is_available():
    print("WARNING: PyTorch can't see the GPU, so this will run on the CPU and be slow.")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {args.model} ...")
tok = AutoTokenizer.from_pretrained(args.model)
# "eager" is the plain softmax(QK^T)V implementation. It is the only one that
# hands back the attention weights; FlashAttention/SDPA never build the full matrix.
model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
cfg = model.config
n_layers, n_heads, n_kv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
group = n_heads // n_kv
print(f"{n_layers} layers, {n_heads} query heads sharing {n_kv} key/value heads")

with open(args.text, encoding="utf-8") as f:
    ids = tok(f.read(), return_tensors="pt").input_ids[0]
need = args.tokens * args.docs
if len(ids) < need:
    raise SystemExit(f"The text has only {len(ids)} tokens; need {need}.")
print(f"Text has {len(ids):,} tokens; using {args.docs} documents of {args.tokens:,}.")

T, L = args.tokens, args.last
q_pos = torch.arange(T - L, T, device=device)                  # query positions measured
k_pos = torch.arange(T, device=device)
dist = q_pos[:, None] - k_pos[None, :]                          # how far back each key is
local_mask = (dist >= 0) & (dist < args.local)

# k90[layer, head, doc*L + i] = tokens needed for 90% of query i's attention
k90 = np.zeros((n_layers, n_heads, args.docs * L), dtype=np.int32)
k50 = np.zeros_like(k90)
sink = np.zeros((n_layers, n_heads, args.docs * L), dtype=np.float32)
local = np.zeros_like(sink)
doc = 0


def make_hook(layer):
    def hook(module, inputs, output):
        attn_out, weights = output[0], output[1]
        if weights is None:
            raise SystemExit("No attention weights came back. Is attn_implementation='eager'?")
        sl = slice(doc * L, (doc + 1) * L)
        for h in range(n_heads):
            w = weights[0, h, -L:, :].float()                   # (L queries, T keys)
            cs = w.sort(dim=-1, descending=True).values.cumsum(dim=-1)
            k90[layer, h, sl] = ((cs < 0.9).sum(-1) + 1).cpu().numpy()
            k50[layer, h, sl] = ((cs < 0.5).sum(-1) + 1).cpu().numpy()
            sink[layer, h, sl] = w[:, 0].cpu().numpy()          # weight on the very first token
            local[layer, h, sl] = (w * local_mask).sum(-1).cpu().numpy()
        # Drop the weights so the next layer doesn't have to keep them in memory.
        return (attn_out, None) + tuple(output[2:])
    return hook


for i, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_hook(make_hook(i))

with torch.no_grad():
    for doc in range(args.docs):
        x = ids[doc * T:(doc + 1) * T].unsqueeze(0).to(device)
        t0 = time.time()
        model(x, use_cache=False)
        print(f"Document {doc + 1}/{args.docs} done in {time.time() - t0:.1f} s")

os.makedirs("results", exist_ok=True)
np.savez_compressed("results/k90.npz", k90=k90, k50=k50, sink=sink, local=local,
                    tokens=T, last=L, local_window=args.local)

# Context available to the measured queries is T-L+1 .. T tokens (about 3,000 to 4,000).
with open("results/per_head.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "head", "kv_head", "k50_median", "k90_median", "k90_p25", "k90_p75",
                "first_token_share", f"last{args.local}_share"])
    for l in range(n_layers):
        for h in range(n_heads):
            v = k90[l, h]
            w.writerow([l, h, h // group, int(np.median(k50[l, h])), int(np.median(v)),
                        int(np.percentile(v, 25)), int(np.percentile(v, 75)),
                        round(float(sink[l, h].mean()), 3), round(float(local[l, h].mean()), 3)])

with open("results/per_layer.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "k90_median", "k90_p25", "k90_p75", "narrowest_head_k90",
                "widest_head_k90", "first_token_share", f"last{args.local}_share"])
    for l in range(n_layers):
        heads = np.median(k90[l], axis=1)
        w.writerow([l, int(np.median(k90[l])), int(np.percentile(k90[l], 25)),
                    int(np.percentile(k90[l], 75)), int(heads.min()), int(heads.max()),
                    round(float(sink[l].mean()), 3), round(float(local[l].mean()), 3)])

print()
print(f"Tokens needed for 90% of the attention (median over heads and the last {L} positions):")
for l in range(n_layers):
    print(f"  layer {l:2d}: {int(np.median(k90[l])):5d}   "
          f"first token gets {sink[l].mean():.0%}, last {args.local} tokens get {local[l].mean():.0%}")
print(f"\nOverall median: {int(np.median(k90))} of about {T - L // 2:,} available tokens.")
print("Saved results/per_head.csv, results/per_layer.csv and results/k90.npz")
