# Explainer check: claim by claim

Checked `explainer.md` (not edited) against the sources in `readings.md`. Status: confirmed / corrected / not found. "Sec." numbers refer to explainer.md sections.

## Corrections needed (summary)

1. Sec. 7 and README: DiffusionGemma does not use a `[MASK]` token. It starts from random vocabulary tokens and re-noises with random tokens.
2. Sec. 7: the 256-token canvas is confirmed, but add the numbers: max 48 denoising steps, adaptive stopping, entropy-bounded sampler.
3. Sec. 4: vocabulary range "150,000 to 250,000" is too low for Gemma (262K).
4. Sec. 5: LLaDA "competitive" needs numbers and the caveat that it trails LLaMA3 8B Instruct after fine-tuning.
5. Sec. 5: MDLM "much closer" should say how close: 14% (LM1B) and a bigger gap on OWT.
6. Sec. 6: SEDD masking claim not confirmed; add real results (25-75%, 32x fewer evaluations).
7. Sec. 8: "Easy text finishes in fewer rounds" is confirmed by Google; add the 4x speed caveat (low batch only).
8. Sec. 3: link to diffusion can now be stated exactly (Proposition 1, Theorem 22).
9. Talk outline slides 14-15 and README line 52: remove "masked" wording for DiffusionGemma.

## Section 1 and 2 (general diffusion)

- "Network predicts noise, clean image, or score": confirmed (MIT notes: score, denoiser `E[z|x]`, noise predictor, all recoverable from each other; Remark 16).
- "SDE with jitter, ODE without": not directly checked in fetched text; MIT notes sections 2.1-2.2 cover ODE and SDE but only the ODE part was seen. Not found for SDE wording.

## Section 3 (flow matching)

- Path `x_t = (1-t) x0 + t x1`, velocity `x1 - x0`: consistent with MIT Eq. 15/37 (`alpha_t z + beta_t eps`, target `alpha_dot z + beta_dot eps`) with alpha_t = t, beta_t = 1-t. Confirmed as a special case (my derivation, not a quoted line). Data is at t=1, noise at t=0, matches the notes.
- Plain MSE loss, no score: confirmed (Eq. 26, "conditional flow matching loss"; Cambridge blog "just performing regression").
- Euler step `x <- x + v/steps`: confirmed (`X_{t+h} = X_t + h u_t(X_t)`, `h = 1/n`).
- Loss does not reach zero, network learns the average: partly confirmed. Cambridge: marginal field is the average of conditional fields, `u_t(x) = E[u_t(x|x1)]`. MIT Theorem 12: marginal and conditional losses are equal up to a constant, so a non-zero floor is expected.
- One step is blurry because it uses the average at t=0: consistent with the above; Cambridge adds that crossing paths make marginal paths curved, which is why more steps help. Corrected wording suggestion: "Crossing conditional paths make the averaged field curved, so one straight Euler step cannot follow it" (Cambridge blog).
- "Link to diffusion ... To verify": confirmed, corrected text: "The vector field and the score convert into each other for Gaussian paths: `u_t(x) = a_t * grad log p_t(x) + b_t * x` (MIT Proposition 1). Learning the denoiser `E[z|x]`, the noise, the score or the velocity are theoretically equivalent." Cambridge: "even exactly the same in some settings".

## Section 4 (why not text)

- Vocabulary "about 150,000 to 250,000": corrected. DiffusionGemma's vocabulary is 262K (model card), so write "roughly 100,000 to 260,000 (DiffusionGemma: 262K)". The lower end is not from a fetched source; keep it vague.
- Continuous diffusion on embeddings "works poorly": not found. The survey text mentions continuous diffusion LMs but gives no verdict. Soften to "the field mostly moved to discrete" or drop. MIT notes say text is "typically modeled as a discrete object".

## Section 5 (masked diffusion, MDLM, LLaDA)

- Forward process: each token masked independently, probability rises from 0 to 1: confirmed (MDLM Sec. 3.2.1; LLaDA: `t ~ U[0,1]`, mask each token with probability t).
- Training = BERT-like with random mask rate and weighted loss: confirmed. MDLM: "weighted average of masked language modeling losses", weight `alpha_t'/(1-alpha_t)`; LLaDA: loss with factor `1/t` on masked tokens only.
- Bidirectional attention: confirmed (LLaDA "does not use a causal mask"; Raschka).
- Generation: start all masked, keep some, repeat; "most confident" example: confirmed for LLaDA (low-confidence remasking keeps the most confident; theory says random).
- MDLM "much closer to autoregressive": corrected text: "MDLM reaches a perplexity upper bound of 23.00 against 20.86 for an autoregressive model on LM1B (about 14% worse), and 23.21 against 17.54 on OpenWebText. The paper says 'within 15-25%'." (MDLM paper Tables 1 and 2.)
- LLaDA "roughly 8B from scratch, competitive": confirmed and add: trained from scratch on 2.3T tokens; the base model is comparable with LLaMA3 8B Base (MMLU 65.9 vs 65.4, GSM8K 70.3 vs 48.7, but HellaSwag 70.5 vs 79.1); after SFT it trails LLaMA3 8B Instruct slightly (MMLU 65.5 vs 68.4). Also note LLaDA cannot use a KV cache (paper).

## Section 6 (SEDD)

- Ratios replace the score, "score entropy" loss: confirmed. Corrected text: "SEDD learns the ratios `p(y)/p(x)` (the concrete score) with the loss `sum_{y != x} w_xy ( s(x)_y - (p(y)/p(x)) log s(x)_y + K(p(y)/p(x)) )`, `K(a) = a(log a - 1)`" (SEDD Eq. 5).
- "Early result that discrete diffusion can be competitive": confirmed and quantified: 25-75% lower perplexity than earlier diffusion LMs, beats GPT-2 zero-shot perplexity, 6-8x better generative perplexity than un-annealed GPT-2, similar quality with 32x fewer evaluations (abstract).
- "Its framework covers masking as one special case": not found in fetched text (the paper was cut off in Sec. 3.2). Keep "to verify" or drop.

## Section 7 (block diffusion, DiffusionGemma)

- Pure masked diffusion has fixed length and no KV cache: confirmed for LLaDA (fixed length hyperparameter; "incompatible with KV caching"). MDLM partly counters this with semi-autoregressive decoding of arbitrary length, so say "the basic version".
- Block recipe (blocks left to right, diffusion inside, cache earlier blocks, variable length): confirmed (Block Diffusion paper: one block at a time given earlier blocks, cached keys and values, parallel sampling within a block; abstract: flexible length). Add: block sizes 4, 8, 16 tested there; smaller blocks gave better perplexity (OWT 20.73 at L'=4 vs MDLM 22.98).
- Canvas of 256 tokens: confirmed (model card: canvas length 256).
- "All 256 positions start masked": corrected. Text for explainer: "The canvas starts as random placeholder tokens from the vocabulary. DiffusionGemma uses uniform-state diffusion, with no mask token. A token the model loses confidence in is replaced by a fresh random token, so earlier choices can be corrected." (Google AI docs 'Diffusion in Text Generation Explained'; Google developer guide; NVIDIA recipe says "there is no [MASK] token".)
- Steps per block: now found. Maximum 48 denoising steps; temperature decays 0.8 to 0.4; entropy-bounded sampler picks lowest-entropy tokens under bound 0.1; the rest are fully renoised; adaptive stop when average canvas entropy < 0.005 and top predictions are unchanged for two steps (model card). A Sudoku demo used 12 steps with early stopping (developer guide).
- How it fits with the cache: confirmed. A causal encoder prefills the prompt and the cache; each finished canvas is appended to the cache; the decoder is bidirectional over the canvas. Add that DiffusionGemma is encoder-decoder, not one decoder.
- "DiffusionGemma = block diffusion": partly confirmed. The docs call it "block-autoregressive multi-canvas sampling (block diffusion)", but the model card cites no papers, so do not say it is the Block Diffusion paper's BD3-LM method. Say "the same idea".
- Extra facts worth adding: 25.2B total / 3.8B active, 8 of 128 experts plus 1 shared, 256K context, Apache 2.0, released 2026-06-10.

## Section 8 (comparison table)

- Parallel, fixed number of runs per block: confirmed ("15-20 tokens per forward pass").
- Fewer rounds on easy text: confirmed by Google ("simpler or structured prompts such as code need fewer denoising steps"; adaptive stopping). Real speed: "up to 4x", over 1,000 tokens/s per user on one H100, over 700 on an RTX 5090, strongest at low to medium batch; may not accelerate on unified-memory Macs (Google blog). Raschka's figure "20-60 steps vs 2,000" is a theoretical argument only.
- Bidirectional context, planning: confirmed (developer guide: non-sequential problems like Sudoku).
- Caching harder, block diffusion restores it: confirmed (LLaDA no KV cache; Block Diffusion and DiffusionGemma cache earlier blocks).
- Quality gap on hard reasoning: confirmed with numbers. DiffusionGemma vs Gemma 4 26B A4B: MMLU Pro 77.6 vs 82.6; AIME 2026 69.1 vs 88.3; GPQA Diamond 73.2 vs 82.3; LiveCodeBench v6 69.1 vs 77.1; MRCR v2 8-needle 128k 32.0 vs 44.1; HLE no tools 11.0 vs 8.7 is the only lead. Google says quality "is lower than standard Gemma 4". The survey text: "significant gaps remain". Raschka: "solid models but not SOTA".
- "Can revise within the block if the sampler allows remasking": confirmed for DiffusionGemma (renoising, self-correction). Add Raschka's downsides: no streaming, tool calling and chain-of-thought are awkward.
- "What to measure in Task 11": steps per canvas can be compared with the 48 maximum; tokens/s vs Gemma 4 26B A4B on the same PC. Google's numbers are for H100/RTX 5090, not this PC.

## To verify list in explainer.md (last section)

- Flow matching vs diffusion relation: confirmed (above).
- MDLM objective and results; LLaDA size and claims: confirmed with numbers (above).
- SEDD definition and results: confirmed; masking special case not found.
- DiffusionGemma block size, steps, sampling: found (256, 48, entropy bound 0.1); Fast-dLLM paper not read.
- Quality gap in survey and Raschka: confirmed; the survey fetch looked like the Dream 7B paper, so re-check arXiv 2508.15487.
