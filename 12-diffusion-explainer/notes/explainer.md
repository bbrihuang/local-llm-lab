# How diffusion works: from images to text

First draft written from background knowledge, before doing the readings. Check each paper-specific claim against the paper (see "To verify" at the end).

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
- **Link to diffusion:** diffusion with a certain noise path and flow matching with Gaussian paths are closely related; the network can be reinterpreted between predicting noise, data, score or velocity. (To verify: the exact equivalence statement in the MIT notes.)

## 4. Why text can't use this directly

Images are continuous numbers, so "a little noise" makes sense. A sentence is a sequence of tokens from a fixed list (about 150,000 to 250,000 words and pieces). There is no meaningful "half way between the word cat and the word dog". Two ways around it:

1. Do continuous diffusion on word embeddings and round to the nearest word at the end (early diffusion language models; it works poorly compared with the approaches below).
2. Define diffusion directly on discrete tokens: the "noise" is replacing tokens with something else. The dominant choice is a **mask** symbol.

## 5. Masked diffusion language models

- **Forward process:** each token is independently replaced by a special `[MASK]` token with a probability that grows from 0 (clean) to 1 (everything masked) as time goes from 0 to 1.
- **Training:** pick a random masking level, mask that fraction of tokens, and train a transformer (with bidirectional attention: every position sees all others) to predict the original tokens at the masked positions. This is a masked language model (like BERT) with a randomized masking rate and a loss weighted by the rate.
- **Generation:** start with every position masked (for a chosen length). In each round, the model predicts all masked positions at once; keep (unmask) some of the predictions and leave the rest masked; repeat until nothing is masked. Which ones to keep is a design choice (for example, keep the most confident ones).
- **MDLM** ("Simple and Effective Masked Diffusion Language Models") showed that with a good training objective and recipe, masked diffusion can come much closer to autoregressive models on perplexity. **LLaDA** ("Large Language Diffusion Models") trained a roughly 8-billion-parameter masked diffusion model from scratch and reported it competitive with similar-size autoregressive models on several benchmarks. (To verify: sizes and claims.)

## 6. Score entropy (SEDD)

Continuous diffusion needs the score (a gradient), which doesn't exist for discrete tokens. SEDD ("Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution") replaces it with **ratios**: how much more likely is sequence y than a neighboring sequence x that differs in one token. A "score entropy" loss trains the network to estimate those ratios. It was an early result showing discrete diffusion can be competitive for language, and its framework covers masking as one special case. (To verify.)

## 7. Block diffusion (DiffusionGemma)

Pure masked diffusion has two practical problems: a fixed output length, and no key/value cache reuse because every position attends to every other. **Block diffusion** is a middle road:

- Split the output into blocks (the Task 11 guide calls each one a canvas of 256 tokens).
- Generate block by block, left to right, like an autoregressive model across blocks.
- Inside one block, use diffusion: all 256 positions start masked and are filled in over several denoising steps.

That keeps parallel generation inside a block, lets earlier blocks be cached, and allows output of any length by adding blocks. (To verify with the Block Diffusion paper and the DiffusionGemma model card: block size, how many steps per block, and how sampling settings work.)

## 8. Diffusion language models against autoregressive ones

| | Autoregressive | Diffusion LM |
|---|---|---|
| Generation order | One token at a time, left to right | All positions of a block at once, over several rounds |
| Cost of an answer | One model run per token | A fixed number of runs per block, however many tokens it holds |
| Easy text | Still one run per token | Can finish in fewer rounds when many tokens are obvious (to verify: how much of this is real in practice) |
| Context | Only what's to the left | Bidirectional inside the block, so it can plan ahead and fill gaps |
| Caching | Standard key/value cache | Harder; block diffusion restores it between blocks |
| Quality | The reference | Closing the gap but generally still behind on hard reasoning and long outputs (to verify against the survey) |
| Control | Cannot revise earlier tokens | Can revise within the block if the sampler allows remasking |

**What to measure in Task 11:** tokens per second against the regular Gemma 4 model of the same size, and how many denoising steps each block used.

## To verify when doing the readings

- The exact relationship between flow matching and diffusion (MIT 6.S184 notes, lectures 1, 2 and 5).
- MDLM's training objective and results; LLaDA's size, data and benchmark claims.
- SEDD's score entropy definition and its results.
- DiffusionGemma specifics (block size, steps per block, sampling settings) from its model card and the Block Diffusion and Fast-dLLM papers.
- The state of the quality gap in the diffusion language model survey and in Raschka's "Beyond Standard LLMs".
