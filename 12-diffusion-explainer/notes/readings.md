# Readings: what each source says (diffusion explainer)

Read on 2026-10-17 through WebFetch (a small model summarizes each page, so wording is the tool's, equations as it reported them). Figures below are only ones seen in fetched text. "Not found" means not in what came back, not that the source lacks it.

## 1. MIT 6.S184 notes (arXiv 2506.02070) and 2026 course page

- Fetched the PDF route (ar5iv returned only the first part, up to Section 2.1). Flow model setup: start `X_0 ~ p_init = N(0, I_d)`, follow `dX_t/dt = u_t^theta(X_t)`, goal `X_1 ~ p_data`. Euler update: `X_{t+h} = X_t + h * u_t(X_t)`, `h = 1/n`. Same as the demo's `x <- x + v/steps`.
- Gaussian conditional path (Eq. 15): `p_t(.|z) = N(alpha_t z, beta_t^2 I)`, with `alpha_0 = beta_1 = 0` and `alpha_1 = beta_0 = 1`. Note the notes put data at t=1 and noise at t=0, like the demo. The straight line `(1-t) x0 + t x1` is the special case alpha_t = t, beta_t = 1-t (my reading, not a quoted line).
- Conditional flow matching loss (Eq. 26): `E || u_t^theta(x) - u_t^target(x|z) ||^2`. For Gaussian paths (Eq. 37) the target is `alpha_dot_t z + beta_dot_t eps`, with `x = alpha_t z + beta_t eps`.
- Theorem 12: the marginal and conditional flow matching losses are equal up to a constant, and their gradients coincide. The "loss never reaches zero because the network learns the average of many pairs" reasoning was not stated in the fetched text (the Cambridge blog does say the marginal field is an average).
- Proposition 1: the vector field and the score convert into each other, `u_t(x) = a_t grad log p_t(x) + b_t x`, with `a_t = beta_t^2 alpha_dot_t / alpha_t - beta_dot_t beta_t` and `b_t = alpha_dot_t / alpha_t`. The "denoiser" `D_t(x) = E[z | x]` can be computed from the vector field; the notes say learning D_t and learning u_t are "theoretically equivalent".
- Text: "A notable exception ... is text data, which is typically modeled as a discrete object." Section 7 covers discrete diffusion with continuous-time Markov chains (CTMCs). Section 7 itself was not seen.
- Course page (diffusion.csail.mit.edu/2026/index.html): lectures have slides AND YouTube video, plus shared course notes and three Colab labs. Lecture 1 "Flow and Diffusion Models", lecture 2 "Flow Matching", lecture 5 "Discrete Diffusion Models" (CTMCs, sampling and training them). So lecture content is not video only. I did not open the slides or videos.

## 2. Cambridge MLG blog, "An Introduction to Flow Matching"

- Flow matching is described as combining Continuous Normalising Flows and diffusion models; training is simulation-free regression of a vector field.
- Transport (continuity) equation: `dp_t/dt = - div( u_t p_t )`. Marginal field is the average of conditional fields: `u_t(x) = E_{x1 ~ p(1|t)} [ u_t(x | x1) ]`.
- Gaussian conditional path `p_t(x|x1) = N(mu_t(x1), sigma_t(x1)^2 I)`, with field `u_t(x|x1) = (sigma_dot/sigma)(x - mu_t) + mu_dot_t`.
- CFM loss is a squared error between `u_theta(t, x)` and `u_t(x|x1)`; same gradients as the marginal loss.
- On diffusion: the post leaves it out and says the two are "deeply related and even exactly the same in some settings", promising a follow-up.
- Sampling still means solving an ODE. Drawbacks named: crossing conditional paths give curved marginal paths (slower sampling) and high loss variance. Fix proposed: mini-batch optimal transport pairing.

## 3. Raschka, "Beyond Standard LLMs" (diffusion section)

- Main selling point: generate several tokens in parallel. His own example: for a 2,000-token answer, "20-60 diffusion steps may be cheaper than the 2000 inference steps" of an autoregressive model, "in theory".
- Training: text is corrupted by random masking and the model predicts the masked tokens; bidirectional, like BERT. LLaDA 8B (Llama 3 style architecture) used as the demo.
- Gemini Diffusion quote: "Generates content significantly faster than even our fastest model so far"; benchmark "on par with their fast Gemini 2.0 Flash-Lite model".
- Quality caveat from ParallelBench: diffusion LLMs under parallel decoding "can suffer dramatic quality degradation". Parallel sampling can give incoherent phrases (his example "New City"). He calls them "solid models but not SOTA".
- Downsides listed: no streaming, chain-of-thought may not help, tool calling is tricky. Fewer steps means faster but worse. Possible fit: small on-device models.

## 4. MDLM, Sahoo et al. 2024 (arXiv 2406.07524)

- Forward process: `q(z_t | x) = alpha_t x + (1 - alpha_t) m`, `m` the one-hot mask vector (each token stays or jumps to mask).
- Objective: the continuous-time NELBO is a weighted average of masked-language-modeling losses; weight `alpha_t' / (1 - alpha_t)`. Loss is invariant to the form of alpha_t.
- SUBS parameterization: mask logit set to minus infinity (the model never predicts the mask), and unmasked tokens are copied unchanged.
- Perplexity (diffusion numbers are upper bounds): LM1B, 327B tokens: MDLM <= 23.00 vs retrained AR 20.86 ("within 14%"). OWT, 262B tokens: AR 17.54 vs MDLM <= 23.21. Intro says "within 15-25%".
- Speed: no time conditioning allows caching denoiser outputs so the sampler can skip steps (SEDD cannot); semi-autoregressive 2048-token decoding 89.3 s vs 2473.9 s for SSD-LM (about 25-30x faster).
- Abstract: encoder-only models with efficient samplers, including semi-autoregressive arbitrary-length text.

## 5. SEDD, Lou et al. (arXiv 2310.16834)

- Score entropy (Definition 3.1, Eq. 5): `L_SE = E_{x~p} sum_{y != x} w_xy ( s_theta(x)_y - (p(y)/p(x)) log s_theta(x)_y + K(p(y)/p(x)) )`, `K(a) = a (log a - 1)`; `K` makes the loss non-negative.
- The ratios `p_t(y)/p_t(x)` are called the "concrete score" (Meng et al. 2022) and generalize the score to discrete spaces. The abstract calls score entropy a loss that "naturally extends score matching to discrete spaces".
- Abstract claims: reduces perplexity by 25-75% vs existing language diffusion models; beats GPT-2 zero-shot perplexity; 6-8x better generative perplexity than un-annealed GPT-2; "similar quality with 32x fewer network evaluations"; controllable infilling.
- Whether masking (absorbing state) is a special case: not found in the fetched part (the text was cut off in Section 3.2).

## 6. LLaDA (arXiv 2502.09992)

- 8B parameters, trained from scratch on 2.3 trillion tokens, 0.13 million H800 GPU hours, SFT on 4.5 million pairs.
- Forward process: sample `t ~ U[0,1]`, mask each token independently with probability t. Loss: `-E[ (1/t) sum_i 1[x_t^i = M] log p_theta(x_0^i | x_t) ]`, on masked tokens only; upper-bounds negative log-likelihood. No causal mask.
- Sampling: theory says random remasking; they use low-confidence remasking. Steps and length are hyperparameters.
- Base vs LLaMA3 8B Base: MMLU 65.9 vs 65.4; GSM8K 70.3 vs 48.7; BBH 49.7 vs 62.1; HellaSwag 70.5 vs 79.1. Instruct vs LLaMA3 8B Instruct: MMLU 65.5 vs 68.4; GSM8K 69.4 vs 78.3. So "competitive", not uniformly better.
- Not KV-cache compatible; used plain multi-head attention instead of grouped-query attention.

## 7. Survey on Diffusion Language Models (arXiv 2508.15487)

- Caution: the fetched text looked like the Dream 7B paper's content, not a classic survey; it may be a different paper or a mis-fetch. Treat as weak.
- Says "significant gaps remain in achieving performance parity with state-of-the-art autoregressive models". Dream 7B vs Qwen2.5 7B: MMLU 69.5 vs 71.9; GSM8K (after SFT) 81.0 vs 91.6; Sudoku 81.0 vs 21.0.
- Says combining diffusion and autoregression "through block generation has emerged as a promising approach" (cites Ye et al. 2024, Arriola et al. 2025). Taxonomy: continuous, discrete, masked.
- KV cache discussion: not found.

## 8. Block Diffusion (arXiv 2503.09573) and DiffusionGemma sources

- Block diffusion: L tokens split into `B = L/L'` blocks of length L'. One block is sampled at a time given earlier blocks; keys and values of earlier blocks are cached; tokens inside a block are sampled in parallel. Abstract: flexible-length generation, KV caching, parallel token sampling.
- Likelihood factorizes: `log p(x) = sum_b log p(x^b | x^{<b})`, each block a discrete diffusion. Block sizes tested: L' = 4, 8, 16 (also 128 in a schedule study). OWT: AR 17.54, MDLM <= 22.98, BD3-LM L'=4 <= 20.73. Up to 13% better perplexity than MDLM on LM1B.
- DiffusionGemma 26B-A4B (Google AI docs, HF card; released 2026-06-10, Apache 2.0): 25.2B total, 3.8B active, 30 layers, 8 of 128 experts plus 1 shared, vocab 262K, context up to 256K. Canvas length 256. Encoder-decoder: autoregressive encoder caches the prompt, bidirectional decoder denoises the canvas, then the finished canvas is appended to the cache.
- Noise is NOT a mask: "Uniform State Diffusion", tokens replaced with random vocabulary tokens; no `[MASK]` token. Training objective: corruption level `t ~ U(eps, 1)`, supervised positions replaced independently by uniform random tokens; optional self-conditioning.
- Recommended sampling: Entropy-Bounded Denoising with adaptive stopping; max 48 denoising steps; temperature decays 0.8 to 0.4; pick lowest-entropy tokens under entropy bound 0.1; non-selected tokens fully renoised; stop when average canvas entropy < 0.005 and top predictions unchanged for two steps.
- Speed: "15-20 tokens per forward pass"; over 1100 tokens/s per user at low batch (H100, FP8); "up to 4x faster"; over 700 tokens/s on RTX 5090. Quality lower than Gemma 4 26B A4B (for example MMLU Pro 77.6 vs 82.6, AIME 2026 69.1 vs 88.3); only HLE no tools is higher (11.0 vs 8.7). Marked experimental.
- Model card cites no papers, so DiffusionGemma's link to the Block Diffusion paper is the docs' own wording ("block-autoregressive multi-canvas sampling"), not a citation. Fast-dLLM paper: not fetched.

## To verify

- MIT notes Section 7 (discrete diffusion, CTMCs) and the explicit straight-line (alpha_t = t) example: not seen.
- SEDD: whether the absorbing/mask process is a special case: not seen.
- The survey link (2508.15487): content did not look like a survey; re-check the arXiv ID.
- Time to first token and wall-clock comparison of DiffusionGemma vs Gemma 4 on this PC: not in any source; Task 11 must measure.
- Fast-dLLM paper and Gemini Diffusion details: not read.
- MIT lecture slides and videos: exist, not opened.
