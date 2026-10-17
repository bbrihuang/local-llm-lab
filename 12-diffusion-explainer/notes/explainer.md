# How diffusion works: from images to text

First drafted from background knowledge, then checked against the sources on 2026-10-17 by Claude (see `explainer_check.md` and `readings.md`). Figures come from fetched page summaries, so recheck the ones you quote.

## 1. The idea in one paragraph

Take real data (an image, a sentence). Destroy it step by step until nothing is left but noise. Train a network to undo one step of that destruction. To generate something new, start from pure noise and apply the network again and again until a clean sample appears. The model never writes the answer left to right: it starts with a whole rough answer and refines all of it.

## 2. Continuous diffusion (images)

- **Forward process:** add a little Gaussian noise at each step. After enough steps the image is indistinguishable from random noise.
- **What the network learns:** given a noisy image and how noisy it is, predict the noise that was added (or, equivalently, the clean image, or the "score": the direction in which the data becomes more likely).
- **Sampling:** start from noise and subtract the predicted noise a little at a time, in tens to hundreds of steps. Doing it with random jitter at each step is a stochastic differential equation (SDE); doing it without jitter is an ordinary differential equation (ODE) that follows the same flow.

## 3. Flow matching (what the 2D demo does)

Flow matching is a simpler way to get the same kind of model.

- **Path:** pick a noise point x0 and a data point x1 and define the straight line `x_t = (1 - t) * x0 + t * x1`.
- **Target:** along that line the velocity is constant: `x1 - x0`.
- **Training:** a network `v(x, t)` regresses on that velocity. The loss is a plain mean squared error. No noise schedule to tune, no score.
- **Sampling:** solve the ODE `dx/dt = v(x, t)` from t=0 to t=1. The simplest solver is Euler steps: `x <- x + v(x, t) / steps`.
- **Why the loss doesn't reach zero:** many (x0, x1) pairs pass through the same (x, t), so the network can only predict their average velocity. The learned field is that average.
- **Why one step is bad:** one Euler step uses the average velocity at t=0 for the whole trip, so everything lands near the middle of the data. More steps let the direction change along the way (see `results/flow_steps.png`).
- **Link to diffusion:** for Gaussian paths the velocity field and the score convert into each other (MIT 6.S184 notes, Proposition 1), so learning the denoiser, the noise, the score or the velocity are theoretically equivalent.
- **Why one step is bad (second reason):** the straight conditional paths cross, so the averaged field is curved and one straight step cannot follow it (Cambridge blog).

## 4. Why text can't use this directly

Images are continuous numbers, so "a little noise" makes sense. A sentence is a sequence of tokens from a fixed list (roughly 100,000 to 260,000 words and pieces; DiffusionGemma's is 262K). There is no meaningful "half way between the word cat and the word dog". Two ways around it:

1. Do continuous diffusion on word embeddings and round to the nearest word at the end (early diffusion language models; the field has mostly moved to the discrete approaches below).
2. Define diffusion directly on discrete tokens: the "noise" is replacing tokens with something else. The dominant choice is a **mask** symbol.

## 5. Masked diffusion language models

- **Forward process:** each token is independently replaced by a special `[MASK]` token with a probability that grows from 0 (clean) to 1 (everything masked) as time goes from 0 to 1.
- **Training:** pick a random masking level, mask that fraction of tokens, and train a transformer (with bidirectional attention: every position sees all others) to predict the original tokens at the masked positions. This is a masked language model (like BERT) with a randomized masking rate and a loss weighted by the rate.
- **Generation:** start with every position masked (for a chosen length). In each round, the model predicts all masked positions at once; keep (unmask) some of the predictions and leave the rest masked; repeat until nothing is masked. Which ones to keep is a design choice (for example, keep the most confident ones).
- **MDLM** ("Simple and Effective Masked Diffusion Language Models") showed that with a good training objective and recipe, masked diffusion gets close to autoregressive models on perplexity: an upper bound of 23.00 against 20.86 on LM1B (about 14% worse) and 23.21 against 17.54 on OpenWebText. **LLaDA** ("Large Language Diffusion Models") trained an 8-billion-parameter masked diffusion model from scratch on 2.3 trillion tokens. The base model is comparable with LLaMA3 8B Base (MMLU 65.9 against 65.4), but after fine-tuning it trails LLaMA3 8B Instruct slightly (65.5 against 68.4). It cannot use a KV cache.

## 6. Score entropy (SEDD)

Continuous diffusion needs the score (a gradient), which doesn't exist for discrete tokens. SEDD ("Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution") replaces it with **ratios**: how much more likely is sequence y than a neighboring sequence x that differs in one token. A "score entropy" loss trains the network to estimate those ratios. It was an early result showing discrete diffusion can be competitive for language: 25% to 75% lower perplexity than earlier diffusion language models, and similar quality with 32 times fewer network evaluations. (Whether masking is a special case of its framework was not confirmed.)

## 7. Block diffusion (DiffusionGemma)

Basic masked diffusion (LLaDA) has two practical problems: a fixed output length, and no key/value cache reuse because every position attends to every other. **Block diffusion** is a middle road:

- Split the output into blocks (the Task 11 guide calls each one a canvas of 256 tokens).
- Generate block by block, left to right, like an autoregressive model across blocks.
- Inside one block, use diffusion over several denoising steps.

That keeps parallel generation inside a block, lets earlier blocks be cached, and allows output of any length by adding blocks.

**What DiffusionGemma actually does** (Google's model card and docs; it cites no papers, so it is "the same idea" as block diffusion, not necessarily the Block Diffusion paper's method):

- **No mask token.** The canvas of 256 tokens starts as random tokens from the vocabulary (uniform-state diffusion). A token the model is unsure about is replaced by a fresh random token, so earlier choices can be corrected.
- **Up to 48 denoising steps** per canvas, with an adaptive stop when the model's uncertainty is low and its top predictions stop changing. Sampling temperature falls from 0.8 to 0.4, and an entropy-bounded sampler (bound 0.1) picks the most certain tokens each step.
- **Encoder-decoder:** a causal encoder reads the prompt and builds the cache; a bidirectional decoder works on the canvas; each finished canvas is appended to the cache.
- 25.2B parameters in total, 3.8B active (8 of 128 experts plus 1 shared), 256K context, Apache 2.0, released 2026-06-10.

## 8. Diffusion language models against autoregressive ones

| | Autoregressive | Diffusion LM |
|---|---|---|
| Generation order | One token at a time, left to right | All positions of a block at once, over several rounds |
| Cost of an answer | One model run per token | A fixed number of runs per block, however many tokens it holds |
| Easy text | Still one run per token | Can finish in fewer rounds when many tokens are obvious (Google: code and structured text need fewer steps) |
| Context | Only what's to the left | Bidirectional inside the block, so it can plan ahead and fill gaps |
| Caching | Standard key/value cache | Harder; block diffusion restores it between blocks |
| Quality | The reference | Still behind: DiffusionGemma against Gemma 4 26B A4B scores AIME 2026 69.1 against 88.3 and MMLU Pro 77.6 against 82.6; it leads only on HLE (11.0 against 8.7) |
| Control | Cannot revise earlier tokens | Can revise within the block if the sampler allows remasking |

**Speed claims:** Google reports up to 4x faster, over 1,000 tokens/s per user on one H100, strongest at low to medium batch size. These are Google's numbers on H100 and RTX 5090, not measured here.

**What to measure in Task 11:** tokens per second against the regular Gemma 4 model of the same size, and how many denoising steps each block used.

## Still to verify

- MIT notes section 7 (discrete diffusion) and the exact SDE wording were not seen.
- Whether masking is a special case of SEDD was not found.
- Fast-dLLM and Gemini Diffusion were not read.
- The survey ID arXiv 2508.15487 fetched as a different paper (looked like Dream 7B); find the right survey before citing it.
- The MIT lecture slides and videos were not opened.
