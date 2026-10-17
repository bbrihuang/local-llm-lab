# Readings for Task 11: what each source says

Read on 2026-10-17 through WebFetch (a small model summarizes each page, so wording is the tool's). Figures are only ones seen in fetched text; "not found" means not in what came back. The Google docs and the Block Diffusion paper are already covered in `../../12-diffusion-explainer/notes/readings.md` (section 8); I reuse them there and only add what is new.

## 1. DiffusionGemma model card (huggingface.co/google/diffusiongemma-26B-A4B-it)

- Built on the Gemma 4 26B A4B mixture-of-experts design, with discrete text diffusion instead of token-by-token output. 25.2B total, 3.8B active, 30 layers, 262K vocabulary, context up to 256K, 8 of 128 experts plus 1 shared. Same as Task 12 notes.
- Encoder-decoder: the encoder prefills the prompt and builds the KV cache; the decoder is bidirectional over the current 256-token canvas. Then the canvas goes into the cache and the next canvas starts.
- Recommended sampling: up to 48 steps, temperature 0.8 to 0.4, entropy bound 0.1, early stop when average canvas entropy is under 0.005 and top predictions are unchanged for two steps.
- Card says "about 15-20 tokens per forward pass" and "above 1,100 tokens/second" per user at low batch on an H100 with FP8 (card's own setup).
- Card says simpler and structured prompts, such as code, need fewer denoising steps, so speed depends on the task.
- Quality is lower than Gemma 4 26B A4B: MMLU Pro 77.6 vs 82.6, AIME 2026 69.1 vs 88.3, GPQA Diamond 73.2 vs 82.3, MMMU Pro 54.3 vs 73.8; only HLE (no tools) is higher, 11.0 vs 8.7.
- Transformers usage: `DiffusionGemmaForBlockDiffusion` plus `AutoProcessor`, `generate` with `max_new_tokens`; vLLM, SGLang and Docker Model Runner are listed for serving.
- Thinking mode is switched on by a `<|think|>` token in the system prompt (the card says this; my demo ran with thinking off).

## 2. Hugging Face blog "Welcome Gemma 4" (huggingface.co/blog/gemma4)

- Found by direct URL; the search did not list it. Family: E2B, E4B, 12B Unified, 31B dense, and 26B A4B (MoE, about 4B active of 26B total, 256K context).
- 26B A4B instruction-tuned scores: MMLU Pro 82.6, AIME 2026 (no tools) 88.3, GPQA Diamond 82.3, LiveCodeBench v6 77.1, MRCR v2 8-needle 128k 44.1. These match the baseline column in the DiffusionGemma card.
- Architecture notes: alternating local sliding-window and global attention, dual RoPE, Per-Layer Embeddings, shared KV cache. No expert counts given.
- Inference engines named: transformers, llama.cpp (GGUF Q4_K_M), mlx-vlm, mistral.rs, transformers.js, ONNX; SGLang has a transformers backend.
- Speed: Multi-Token Prediction drafters (speculative decoding) give "up to ~3x" end-to-end, depending on hardware, batch size and workload.
- No tokens-per-second figure for the plain 26B A4B: not found.
- The blog does not recommend a specific engine for peak speed; it lists several.

## 3. Block Diffusion, Arriola et al. (arXiv 2503.09573)

- Reused from Task 12 notes (section 8): blocks of L' tokens, earlier blocks cached, tokens inside a block sampled in parallel; OWT perplexity AR 17.54, MDLM <= 22.98, BD3-LM L'=4 <= 20.73.
- Re-read through ar5iv (truncated): autoregressive over blocks, discrete diffusion inside each block, interpolating between the two.
- Earlier-block KV caching is presented as what plain bidirectional diffusion cannot do.
- Needs an order of magnitude fewer generation steps than the Gaussian embedding diffusion baseline (Han et al.). No absolute speed figures seen.
- Can generate sequences longer than the training context.
- Link to DiffusionGemma is the docs' wording, not a citation (Task 12 notes).

## 4. Fast-dLLM, Wu et al. (arXiv 2505.22618), via ar5iv

- Training-free speedup for LLaDA and Dream (not for DiffusionGemma, which is a different model).
- Block-wise approximate KV cache: bidirectional attention rules out an exact cache, so it caches prefix keys and values, reuses them within a block and refreshes after each block. Reason given: KV activations at adjacent steps have high cosine similarity.
- DualCache also caches the masked suffix tokens.
- Confidence-aware parallel decoding: unmask every token whose confidence passes a threshold instead of a fixed top-K; if none passes, unmask the single most confident one so decoding always progresses.
- Theorem 1 (as summarized): parallel decoding matches sequential greedy decoding when each marginal confidence is above 1-eps and (n+1)eps <= 1. Error bounds grow with the number of tokens decoded in parallel.
- Reported: up to 27.6x throughput over vanilla LLaDA (8-shot, 1024 tokens), 8.1x on GSM8K (5-shot, 256 tokens), 13.3x for the parallel part alone; single A100, batch 1. Exact accuracy numbers: not found ("minimal loss" only).

## 5. Extra source found: vLLM recipe for DiffusionGemma (recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it)

- Not on my list, but it has the only like-for-like speed comparison I saw. Single H100, concurrency 1, SPEED-Bench, vs Gemma 4 26B-A4B autoregressive: output throughput about 199 to 375 tok/s (1.9x); per-request generation about 205 to 1,282 tok/s (6.2x).
- Time to first token rises from about 53 ms to 489 ms (about 10x); mean request time 2.87 s to 0.88 s.
- Keep `--max-num-seqs` at 4 or less (larger values ran out of memory); needs a vLLM build with diffusion support (0.24.0+).

## What this explains in my results

- 12 to 32 tokens per forward pass. Source says: the model card quotes about 15-20 on average and says code needs fewer steps. My range brackets that; page prompts (24-32) are above it, chat prompts (12-19) are at or below it.
- Pages faster than chat (247/340/265 vs 147/175/146 tok/s). Source says: the card says structured prompts such as code need fewer denoising steps. My guess: the "easy tokens like tags and braces" reason from my README is consistent with this but was not tested.
- Regular Gemma 4 26B-A4B flat at 39.3 tok/s. Source says: on one H100 with vLLM the same baseline ran about 199-205 tok/s, five times my number. My guess: my 39.3 is plain `transformers` overhead (MoE routing and Python loop), as the README says; this supports it but is a different GPU and engine, so not proof.
- Speedup 3.7x chat and 6.3-8.6x pages. Source says: vLLM recipe shows 6.2x per-request generation on H100, in the range of my page numbers. My guess: my ratio is inflated by the slow baseline; against a vLLM-class baseline the gap could be nearer 1.9x (the recipe's whole-run throughput ratio) to 6x, depending on how it is counted.
- Diffusion absolute speed 146 to 340 tok/s versus the card's 1,100+. Source says: the card's number is H100 + FP8 + tuned serving. My guess: the vLLM recipe's 1,282 tok/s on BF16 H100 suggests the engine matters more than the GPU; I ran plain `transformers` BF16 on a Blackwell G4 (RTX PRO 6000).
- Quality not measured in my run. Source says: the card shows DiffusionGemma below Gemma 4 26B A4B on nearly every listed benchmark (e.g. AIME 69.1 vs 88.3), so "faster" does not mean "equal".
- Step counts (3 to 20 of 48). Source says: early stopping on low entropy exists in the sampler, so fewer than 48 steps is expected. Why summary needed 20 steps: my guess, prose has more real choices.
- Fast-dLLM: source says it speeds up LLaDA/Dream by caching and by confidence thresholds; this is the same idea as DiffusionGemma's entropy-bounded selection (my guess, the model card cites no papers). It does not explain my numbers directly.

## To verify

- Re-run the regular model with vLLM or torch.compile on the G4 and see whether 39.3 tok/s rises to near 200; then recompute the speedups.
- Measure time to first token for both models; the vLLM recipe says diffusion is about 10x worse (53 ms vs 489 ms), I did not measure it.
- Check whether `max_new_tokens` 1,024 and 4 canvases hide a quality drop; run the Task 9 HumanEval setup on both.
- Open the real HF model card and blog in a browser: both came through a summarizer, and the blog was not found by search.
- Fast-dLLM exact accuracy numbers and the Theorem 1 statement: read the PDF; ar5iv summary gave no accuracy figures.
- Block Diffusion default block size and speed numbers: the visible text was truncated.
