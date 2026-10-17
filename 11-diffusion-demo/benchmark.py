"""Speed comparison: DiffusionGemma against the regular Gemma 4 26B-A4B (Task 11).

Both models are loaded in bf16 on the same GPU, thinking off, same prompts, same token limit.
Time = wall-clock for model.generate() with the GPU synchronised before and after.
Tokens = tokens written after the prompt, up to and including the first end-of-sequence token.

Diffusion also gets one extra, untimed run with a counting streamer to record the denoising steps
used by every 256-token canvas (the streamer copies data off the GPU each step, so it is kept out of the timing).

The regular model uses greedy decoding (deterministic); DiffusionGemma uses its default entropy-bounded sampler.
"""
import csv
import json
import os
import queue
import statistics
import time

import torch

from core import FrameStreamer, build_inputs, count_new_tokens

PROMPTS = [
    # (short name, kind, text)
    ("coffee-page", "page", "A landing page for a small coffee shop with a menu, opening hours and a contact form."),
    ("todo-app", "page", "A to-do list app where I can add, tick off and delete tasks, saved in the browser."),
    ("pomodoro", "page", "A pomodoro timer with start, pause and reset buttons and a progress ring."),
    ("explain", "chat", "Explain in plain language how a hash table works and why lookups are fast."),
    ("python-fn", "chat", "Write a Python function that merges overlapping intervals, with a short docstring and two examples."),
    ("summary", "chat", "Give me a 6-step plan for learning to cook ten simple dinners, with one tip per step."),
]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _generate(kind, model, inputs, max_new_tokens, streamer=None):
    with torch.inference_mode():
        if kind == "diffusion":
            return model.generate(**inputs, max_new_tokens=max_new_tokens, streamer=streamer)
        return model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)


def run(kind, model, processor, max_new_tokens=1024, repeats=3, out_dir="results", prompts=PROMPTS):
    """Benchmark one model. Writes results/bench_<kind>.json and returns the rows."""
    os.makedirs(out_dir, exist_ok=True)
    # warm-up so compilation / cache setup is not counted
    inp, _ = build_inputs(processor, model, "Say hello in one sentence.")
    _generate(kind, model, inp, 64)
    _sync()

    rows = []
    for name, ptype, text in prompts:
        inp, plen = build_inputs(processor, model, text, as_page=(ptype == "page"))
        times, toks = [], []
        for _ in range(repeats):
            _sync()
            t0 = time.perf_counter()
            out = _generate(kind, model, inp, max_new_tokens)
            _sync()
            dt = time.perf_counter() - t0
            seq = getattr(out, "sequences", out)
            n = count_new_tokens(seq, plen, model)
            times.append(dt)
            toks.append(n)
        row = {"model": kind, "prompt": name, "type": ptype, "prompt_tokens": plen,
               "new_tokens_median": statistics.median(toks),
               "seconds_median": round(statistics.median(times), 3),
               "tokens_per_s_median": round(statistics.median(n / t for n, t in zip(toks, times)), 1),
               "tokens_per_s_runs": [round(n / t, 1) for n, t in zip(toks, times)]}
        if kind == "diffusion":
            q = queue.Queue()
            st = FrameStreamer(processor.tokenizer, q, keep_text=False)
            out = _generate(kind, model, inp, max_new_tokens, streamer=st)
            row["steps_per_canvas"] = [c["steps"] for c in st.canvases]
            tpf = getattr(out, "tokens_per_forward", None)
            row["tokens_per_forward"] = float(tpf[0] if hasattr(tpf, "__len__") else tpf) if tpf is not None else None
        rows.append(row)
        print(f"{kind:9} {name:12} {row['new_tokens_median']:>6.0f} tokens  {row['tokens_per_s_median']:>7.1f} tok/s"
              + (f"  steps/canvas {row['steps_per_canvas']}" if kind == "diffusion" else ""))
    with open(os.path.join(out_dir, f"bench_{kind}.json"), "w") as f:
        json.dump({"max_new_tokens": max_new_tokens, "repeats": repeats,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "torch": torch.__version__, "rows": rows}, f, indent=1)
    return rows


def combine(out_dir="results"):
    """Read both JSON files, write bench_summary.csv and bench_speed.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {k: json.load(open(os.path.join(out_dir, f"bench_{k}.json"))) for k in ("diffusion", "regular")}
    by = {k: {r["prompt"]: r for r in v["rows"]} for k, v in data.items()}
    names = [p[0] for p in PROMPTS if p[0] in by["diffusion"] and p[0] in by["regular"]]
    with open(os.path.join(out_dir, "bench_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "type", "diffusion_tok_s", "regular_tok_s", "speedup", "diffusion_tokens",
                    "regular_tokens", "diffusion_steps_per_canvas", "diffusion_tokens_per_forward"])
        for n in names:
            d, r = by["diffusion"][n], by["regular"][n]
            w.writerow([n, d["type"], d["tokens_per_s_median"], r["tokens_per_s_median"],
                        round(d["tokens_per_s_median"] / r["tokens_per_s_median"], 2),
                        d["new_tokens_median"], r["new_tokens_median"],
                        " ".join(map(str, d.get("steps_per_canvas", []))), d.get("tokens_per_forward")])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], [by["diffusion"][n]["tokens_per_s_median"] for n in names], 0.38,
           color="#2a6ebb", label="DiffusionGemma")
    ax.bar([i + 0.2 for i in x], [by["regular"][n]["tokens_per_s_median"] for n in names], 0.38,
           color="#d9822b", label="Gemma 4 26B-A4B (regular)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("Tokens per second (median of runs)")
    ax.set_title("Same size, same GPU: diffusion against regular generation")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bench_speed.png"), dpi=150)
    print("Saved bench_summary.csv and bench_speed.png in", out_dir)
