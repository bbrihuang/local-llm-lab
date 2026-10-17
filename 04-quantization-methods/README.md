# Task 4: Quantization methods

**Who did what.** Claude read the sources and drafted the 20-slide talk and the notes (`notes/readings.md`); I have not yet presented it and will skim it first. Claude wrote the code (`quant.py`, `layer_error.py`, `model_ppl.py`) and I ran it on my own computer and pasted the output. I did not watch any videos.

## The question
Papers on LLM quantization (LLM.int8(), SmoothQuant, GPTQ, AWQ, QuaRot, MXFP4/NVFP4, BitNet) each claim a better way to use fewer bits. Which ones actually help, and where?

## What I did
- Read the sources and built a talk: [slides](https://claude.ai/artifact/UJmkYKFN3itjaZYmJAe8kR) (private link; export to PDF for the portfolio).
- Wrote one file, `quant.py`, with 21 quantization methods using **fake quantization**: values are rounded the way the format would round them, then computed in normal floats. This measures quality, not speed.
- `layer_error.py`: quantize 12 layers (2, 14, 26; q, o, up, down) of Qwen3-0.6B and measure the relative output error on WikiText-2.
- `model_ppl.py`: quantize every linear layer and measure WikiText-2 perplexity (unquantized: 19.06).

## Results (Qwen3-0.6B)
Full tables: `results/model_ppl_summary.md`.

| Setting | Method | Perplexity | Change |
|---|---|---|---|
| 8-bit | per-token | 19.35 | +1.5% |
| 8-bit | LLM.int8() | 19.06 | 0% |
| 8-bit | SmoothQuant | 19.14 | +0.4% |
| 8-bit | one scale per batch | 33.10 | +74% |
| 4-bit weights | per channel | 32.83 | +72% |
| 4-bit weights | group 128 | 23.36 | +22% |
| 4-bit weights | GPTQ | 22.06 | +16% |
| 4-bit weights | AWQ | 21.32 | +12% |
| 4-bit weights | NVFP4 | 21.35 | +12% |
| 4-bit weights | MXFP4 | 23.12 | +21% |
| 4-bit weights + activations | int4 per token | 93,157 | collapses |
| 4-bit weights + activations | Hadamard + GPTQ (QuaRot recipe) | 34.49 | +81% |
| 4-bit weights + activations | MXFP4 | 30.14 | +58% |
| 4-bit weights + activations | NVFP4 | 23.85 | +25% |

## What it means
- 8-bit is solved. Outliers are the reason one shared scale fails, and LLM.int8() (keep the big features in 16 bits) is nearly lossless.
- For 4-bit weights, GPTQ and AWQ roughly halve the damage of plain rounding, and NVFP4's small blocks do as well as AWQ with no calibration.
- 4-bit for activations too only worked with NVFP4 here (+25%).
- Against what the papers suggest, Hadamard rotation did **not** help: it hurt weight-only group-128 (27.8 vs 23.4) and FP4. It rescued int4 activations only a little (93,157 to 89) and needed GPTQ to reach 34.5. This may come from the tiny model or from my simple implementation, so I would not generalize it.
- Ternary weights (BitNet) applied after training gives perplexity in the millions. BitNet trains for it, so this only shows why that matters.

## Caveats
One small model, one dataset, fake quantization (no speed), and the layer-error and perplexity runs use different amounts of data. Differences of a few percent are within noise I did not measure.

## Commands
```
cd C:\Users\brian\code\local-llm-lab\04-quantization-methods
..\.venv\Scripts\python layer_error.py
..\.venv\Scripts\python model_ppl.py
```

## Still to do
- Present the talk (about 20 minutes) and record how it went.
- Export the slides to PDF and add it to this folder.
- Check the claims in `notes/readings.md` against the original papers where marked.
