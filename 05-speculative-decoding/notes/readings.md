# Reading notes for Task 5

Summaries of the four assigned readings (plus two extra sources), with page and section references. Use these when you do the reading. Lines marked [Inference] are not stated directly by the source.

## Leviathan, Kalman, Matias - Fast Inference from Transformers via Speculative Decoding (arXiv 2211.17192, v2 dated 2023-05-18)

https://arxiv.org/abs/2211.17192

- Section map: 1 Introduction (p.1-2); 2 Speculative Decoding (p.2-3: 2.1 Overview, 2.2 Standardized Sampling, 2.3 Speculative Sampling); 3 Analysis (p.3-5: 3.1 Number of Generated Tokens, 3.2 Calculating alpha, 3.3 Walltime Improvement, 3.4 Number of Arithmetic Operations, 3.5 Choosing gamma, 3.6 Approximation Models). Outside the assigned reading: 4 Experiments (p.6), 6 Discussion (p.8).
- Sec. 1, p.1: large-model inference is 'often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication', 'so additional computation resources might be available'. The key observation is 'that some inference steps are "harder" and some are "easier"'. Abstract: hard language-modeling tasks often include easier subtasks that more efficient models can approximate well.
- Sec. 2.1, p.2: M_p is 'the target model, inference from which we're trying to accelerate'. M_q is 'a more efficient approximation model for the same task'. M_q proposes gamma tokens and M_p checks them in parallel, accepting some and adding one more token 'if they are all accepted'. 'Each parallel run of the target model Mp will produce at least one new token', so the method never needs more serial M_p runs than ordinary decoding.
- Sec. 2.3, p.3: when a guess is rejected, the replacement token is sampled from p'(x) = norm(max(0, p(x) - q(x))). The output distribution stays exactly that of M_p.
- Sec. 3.1, p.3: Definition 3.1 says the acceptance rate beta_{x<t} is the probability of accepting x_t ~ q(x_t|x<t). 'If we make the simplifying assumption that the betas are i.i.d., and denote alpha = E(beta)', then the number of tokens per run 'is a capped geometric variable, with success probability 1 - alpha and cap gamma + 1'. Eq. (1): E(# generated tokens) = (1 - alpha^(gamma+1)) / (1 - alpha).
- Sec. 3.2, p.3-4: Corollary 3.6 gives alpha = 1 - E(D_LK(p,q)) = E(min(p,q)). In plain words, alpha measures how much the draft's and the target's next-token distributions overlap.
- Sec. 3.3, p.4: c, 'the cost coefficient', is 'the ratio between the time for a single run of M_q and the time for a single run of M_p'. The assumption is 'we can run gamma + 1 concurrent evaluations of Mp in parallel without increasing the walltime'. T is 'the cost of running a single step of Mp', and one run of Algorithm 1 costs Tc*gamma + T. Theorem 3.8 gives the expected walltime improvement factor = (1 - alpha^(gamma+1)) / ((1 - alpha)(gamma*c + 1)). Corollary 3.9: if alpha > c, some gamma gives an improvement of at least (1 + alpha)/(1 + c).
- Sec. 3.4, p.4: 'the total number of arithmetic operations might be higher than that of the standard decoding algorithm'. Theorem 3.11 gives the expected increase factor in operations = (1 - alpha)(gamma*c_hat + gamma + 1) / (1 - alpha^(gamma+1)). Here c_hat is the ratio of arithmetic operations per token of M_q to M_p (Definition 3.10).
- Sec. 3.5, p.5: the best gamma depends on alpha and c and is found numerically (Figure 3).
- Sec. 3.6, p.5: n-gram approximation models cost almost nothing ('a table lookup'). An English-German bigram model gives 'alpha ≈ 0.2 which leads to an inference speed improvement factor of 1.25X with gamma = 3'. Text likely to repeat (e.g., summarization, chat) may suit a model that copies tokens from the context.
- Sec. 4, Tables 2-3, p.6-7 (beyond the assigned sections): on T5-XXL, alpha and speedup are higher with argmax sampling (temp=0) than with temp=1. Examples: EnDe with T5-small, alpha 0.75 / 3.4X at temp=0 vs 0.62 / 2.6X at temp=1; CNNDM with T5-small, 0.65 / 3.1X vs 0.53 / 2.3X.
- Sec. 6 Discussion, p.8 (beyond the assigned sections): 'latency is improved through increased concurrency'. 'Thus, our method is not helpful for configurations where additional computation resources are not available.'

## LLM Inference Handbook - Speculative decoding (BentoML handbook, now served by Modular)

https://handbook.modular.com/inference-optimization/speculative-decoding/

- Why speculative decoding: 'GPUs have massive compute capacity but limited memory bandwidth'; 'Much of their compute sits unused while waiting on memory access'; 'Many next tokens are obvious from context and can be proposed by a smaller model'. It is 'especially useful for latency-sensitive applications like chatbots and code completion tools'.
- Key metrics: acceptance rate alpha, speculative token count gamma, and acceptance length tau = (1 - alpha^(gamma+1)) / (1 - alpha). This matches Leviathan Eq. (1). The page states no i.i.d. assumption.
- How acceptance rate impacts performance: in a simulated BentoML test (patched vLLM), latency and throughput improved almost linearly with alpha. Raising gamma helped only when tau was high. Simulated speedups reached 2-3x at alpha >= 0.6 and gamma >= 5, but 'In practice, however, the speedup was lower than expected'. No reason is given.
- How performance varies under different workloads (Llama-3.3-70B-Instruct, vLLM, H100): with TP=1, throughput 'plateaued earlier (around 20-30 concurrent requests)' while TPOT improved about 2x. With gamma=5, latency spikes appeared at 40+ concurrent requests. The page's only stated reason: 'coordination between the draft and target models might bring overhead at higher loads'. The words 'compute-bound' and 'batch size' do not appear on the page.
- Don't ignore wasted compute: 'When the target model rejects too many draft tokens, your GPU still spends time generating and verifying them'; 'The work doesn't pay off'. DSpark section: verifying low-value tail tokens 'is wasted work, and under heavy load it competes with batch capacity that could serve other requests.'
- Memory overhead: the draft and target models must both fit in GPU memory, which can reduce batch capacity on a single GPU.
- N-gram speculation: copies the continuation that followed an earlier match of the recent suffix. It 'is helpful when the desired output repeats or closely follows language already available in the prompt'. Examples listed: summarizing or rewriting supplied text, editing code or configuration files, filling predefined templates, reproducing names, dates, identifiers or technical terms, and generating structured output from a schema in the prompt. DFlash reports 'the largest gains on math and coding tasks'.
- Get the draft model right: how closely the draft's distribution matches the target 'determines the acceptance rate'. Off-the-shelf drafts 'often struggle with domain-specific tasks' or very long contexts, and fine-tuning the draft on your own data helps.

## Aleksa Gordic - Inside vLLM: Anatomy of a High-Throughput LLM Inference System (2025-09-05; vllm.ai repost of the post from aleksagordic.com/blog/vllm)

https://vllm.ai/blog/2025-09-05-anatomy-of-vllm

- Speculative Decoding, opening: 'In autoregressive generation, each new token requires a forward pass of the large LM. This is expensive - every step reloads and applies all model weights just to compute a single token!' Because 'a single large-model pass can yield up to k+1 tokens', speculation pays off.
- Steps: (1) the draft proposes k tokens; (2) the large model runs once on the context plus k drafts, giving probabilities for k+1 positions; (3) going left to right, accept a token if p_large >= p_draft, otherwise accept with probability p_large/p_draft, and stop at the first rejection. If all k are accepted, the (k+1)-th token comes free from the large model's distribution. On a rejection, resample from (p_large - p_draft) clamped at 0 and normalized. This 'guarantees that in expectation the sequence is distributed exactly as if we had sampled token by token from the large model'.
- vLLM V1 'does not support the LLM draft model method'. It uses n-gram, EAGLE and Medusa proposers. The n-gram proposer takes the last prompt_lookup_max tokens, finds an earlier match in the sequence, and proposes the k tokens that followed. If there is no match, it shrinks the window down to prompt_lookup_min. Example config: method ngram, prompt_lookup_max 5, prompt_lookup_min 3, num_speculative_tokens 3.
- Scheduler section: decode is 'memory-bandwidth-bound, since we still need to load all LLM weights (and KV caches) just to compute one token', while prefill is usually compute-bound.
- Benchmarks and auto-tuning - latency vs throughput (roofline): 'below a saturation batch B_sat, the step time is dominated by HBM bandwidth', 'so step latency is nearly flat'. 'Beyond B_sat, the kernels become compute-bound and step time grows roughly with B'. As B grows, throughput improves 'because weight I/O is amortized across more tokens'. The post does not connect this passage to speculative decoding. That connection is mine (see the 'why speedup shrinks' answer).

## Agrawal et al. - On Evaluating Performance of LLM Inference Serving Systems (arXiv 2507.09019v1, 2025-07-11)

https://arxiv.org/abs/2507.09019

- This is the right paper: the exact title, and it has a speculative decoding case study. Authors: Amey Agrawal, Nitin Kedia, Anmol Agarwal, Jayashree Mohan, Nipun Kwatra, Souvik Kundu, Ramachandran Ramjee, Alexey Tumanov.
- Structure: 2 Background (2.3: 'The prefill phase is compute-bound'; 'the decode phase is memory-bound'); 3 Common Anti-Patterns (p.4-8) in three groups. Baseline fairness: AP1 Conflating Implementation and Algorithm, AP2 Neglecting Parameter Tuning. Evaluation setup: AP3 Outdated or Irrelevant Models, AP4 Non-Representative Workloads, AP5 Ignoring Practical Latency Bounds. Metric design: AP6 Misleading or Opaque Metrics, AP7 Reporting Only Summary Statistics, AP8 Obscuring Performance with Normalization. Then 4 Case Study: Evaluating Speculative Decoding Systems (p.8-9) and 5 Conclusion.
- Case study setup (Figure 3, p.8): 'Speculative decoding vs. autoregressive performance (0.25 QPS, ShareGPT4, Llama-3 70B, 4xH100)'. It uses one low load level only.
- Findings: 'successful verifications yield bursts of tokens arriving almost instantly' while verification failures add delays, so the time-between-tokens distribution 'becomes highly bimodal'. Median TPOT is '1.3x lower in our experiments', but 'this advantage can reverse at the tail (e.g., 1.16x higher P99 TPOT)'. 'Speculative decoding consistently incurs higher TTFT' because of draft-model prefill overhead. The Fluidity Index shows more user-visible stalls.
- The benefit 'depends heavily on factors like the draft model's accuracy and the typical response length', and 'benefits increase for longer generations as the initial prefill overhead is amortized'.
- What it does NOT say: it never analyzes how the speedup changes with load or batch size, and it does not compare acceptance rates across task types (code vs chat, etc.). I could not use it to support the 'shrinks under load' or 'predictable text' points. Its lesson is about measurement: averages hide burstiness and tail latency (AP7, AP8; TTFT ties to AP5).

## SUPPLEMENTARY (not in the assignment): vLLM Team - How Speculative Decoding Boosts vLLM Performance by up to 2.8x (2024-10-17)

https://vllm.ai/blog/2024-10-17-spec-decode

- I added this because the assigned readings only indirectly support the 'shrinks under many users' point.
- 'Speculative decoding offers significant performance benefits in low-QPS (queries per second) environments': up to 1.5x on ShareGPT with a draft model, and up to 2.8x with prompt lookup (n-gram) on CNN/DailyMail summarization.
- 'However, in high-QPS environments, speculative decoding may introduce performance trade-offs. The extra compute required to propose and verify tokens can sometimes slow down the system when it is already compute-bound, as seen when the number of requests per second increases. In such cases, the overhead of speculative decoding can outweigh its benefits'. The figure caption reports 1.4x slowdown (Llama3-70B, ShareGPT, 4xH100) and 1.8x slowdown (Llama3-70B, CNN/DailyMail, 4xH100) at high QPS.
- Prompt lookup 'works particularly well when the large model repeats parts of the prompt in its answers'.
- Roadmap item: 'dynamic speculative decoding shortens the proposed length when system load is high'.

## SUPPLEMENTARY (not in the assignment): Liu et al. - TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput (arXiv 2406.14066v3, 2025-07-27)

https://arxiv.org/html/2406.14066v3

- Sec. 1: 'tokens that fail verification must be discarded, resulting in wasted computational resources'. 'While computation inefficiencies due to incorrect token speculations may be acceptable under low system load, they become prohibitive if the acceptance rate drops or the request rate to the system increases.'
- Sec. 4.1.2: 'increasing the batch size tends to produce more tokens at the same cost'. Large batches (BS=64) do best with short proposals (k=1-2), and BS=1 peaks at k=4-5. Low acceptance (0.3) favors short proposals 'to minimize wasted computation'.
- Sec. 6.1: plain speculative decoding shows 'significant performance deterioration' at batch sizes of 16 or more. TurboSpec avoids this by turning speculation down or off.
- Sec. 3: the acceptance rate 'is dynamically determined by the user request, the LLM model used, and the speculation method'.

## The formulas

Notation, as defined in Leviathan et al.:
- M_p: the target model you want to speed up (Sec. 2.1, p.2).
- M_q: the cheaper approximation (draft) model (Sec. 2.1, p.2).
- gamma: a positive integer, the number of tokens M_q guesses per run (Sec. 2.1, p.2).
- beta: the acceptance rate for a given prefix, i.e. the probability that speculative sampling accepts M_q's proposed token (Definition 3.1, p.3).
- alpha = E(beta): the average acceptance rate, under the paper's 'simplifying assumption that the betas are i.i.d.' (Sec. 3.1, p.3). Corollary 3.6 gives alpha = 1 - E(D_LK(p,q)) = E(min(p,q)), roughly how much the two models' next-token distributions overlap.
- c: 'the cost coefficient', 'the ratio between the time for a single run of M_q and the time for a single run of M_p' (Sec. 3.3, p.4).
- T: 'the cost of running a single step of Mp'. One run of Algorithm 1 costs T*c*gamma + T (proof of Theorem 3.8, p.4).
- c_hat: the ratio of arithmetic operations per token of M_q to that of M_p (Definition 3.10, p.4).

Eq. (1), Sec. 3.1, p.3 (expected tokens produced per run, i.e. per M_p step):
E(# generated tokens) = (1 - alpha^(gamma+1)) / (1 - alpha)
This holds because the count 'is a capped geometric variable, with success probability 1 - alpha and cap gamma + 1'. The result is always at least 1 (Sec. 2.1) and at most gamma + 1. The BentoML/Modular handbook gives the same formula and calls it the acceptance length tau.

Theorem 3.8, Sec. 3.3, p.4 (walltime):
'The expected improvement factor in total walltime by Algorithm 1 is' (1 - alpha^(gamma+1)) / ((1 - alpha)(gamma*c + 1)).
- Reading: Eq. (1) divided by the relative cost of one run, (gamma*c + 1).
- Stated assumptions: i.i.d. betas; 'we can run gamma + 1 concurrent evaluations of Mp in parallel without increasing the walltime'; long enough generations.

Corollary 3.9, p.4: 'If alpha > c, there exists gamma for which we'll get an improvement, and the improvement factor will be at least' (1 + alpha)/(1 + c).

Theorem 3.11, Sec. 3.4, p.4 (expected factor of increase in total arithmetic operations):
(1 - alpha)(gamma*c_hat + gamma + 1) / (1 - alpha^(gamma+1))

Sec. 3.5, p.5: the best gamma depends on alpha and c and is found numerically (Figure 3).

The paper's own check (Sec. 3.6, p.5): a bigram draft with alpha ≈ 0.2, near-zero c and gamma = 3 gives 1.25X. That matches (1 - 0.2^4)/(0.8 x 1) = 1.248.

My worked numbers (computed with c = 0.05):

| alpha | tokens per pass, gamma = 1 / 3 / 5 / 7 | walltime gain, gamma = 1 / 3 / 5 / 7 |
|---|---|---|
| 0.2 | 1.20 / 1.25 / 1.25 / 1.25 | 1.14 / 1.09 / 1.00 / 0.93 |
| 0.5 | 1.50 / 1.88 / 1.97 / 1.99 | 1.43 / 1.63 / 1.57 / 1.48 |
| 0.8 | 1.80 / 2.95 / 3.69 / 4.16 | 1.71 / 2.57 / 2.95 / 3.08 |

So guessing further ahead only pays off when alpha is high.

Caveat: the arXiv PDF's text extraction garbled the fraction layout, so Eq. (1) and Theorem 3.8 were rebuilt and then checked against the handbook's tau formula and the paper's 1.25X example.

## Why predictable text speeds up more

In plain terms: speculative decoding pairs a big, slow, accurate model (the 'target', M_p) with a cheap guesser. The guesser is either a small 'draft' model (M_q) or a lookup that copies text already in the prompt (n-gram / prompt lookup). The guesser writes a few tokens ahead. The big model then checks all of those guesses in a single pass. Every accepted guess is a token you got without paying for a separate big-model step. Every rejected guess is thrown away, and the big model supplies the correct token itself (Leviathan Sec. 2.1, p.2; Gordic, 'Speculative Decoding'). The output is exactly what the big model would have produced on its own (Leviathan Sec. 2.3; Gordic). So speculation can only change how fast you go, never what you get.

The speed therefore depends on how often the guesser is right. Leviathan et al. call this the acceptance rate, alpha (Definition 3.1 and Sec. 3.1, p.3). Corollary 3.6 (p.4) shows alpha = E(min(p, q)), which measures how much the two models' next-token probability distributions overlap. A higher alpha means more tokens per big-model pass (Eq. 1). The BentoML/Modular handbook says the same thing in words: higher alpha gives fewer target passes, lower latency and higher throughput. The intuition comes from Leviathan Sec. 1, p.1: 'some inference steps are \"harder\" and some are \"easier\"', and the handbook adds that 'Many next tokens are obvious from context and can be proposed by a smaller model.'

Why predictable text wins:
1. Rewriting code, editing config files, filling templates, copying names and identifiers, and summarizing a supplied document all mostly repeat or closely follow text already in the prompt. In those cases even a free 'copy from the prompt' guesser is usually right. The handbook's n-gram section lists exactly these cases: 'helpful when the desired output repeats or closely follows language already available in the prompt' (examples: editing code or configuration files, filling predefined templates, reproducing identifiers, structured output from a schema). Leviathan Sec. 3.6 (p.5) makes the same point about text likely to repeat. The supplementary vLLM 2024 post reports up to 2.8x speedup from prompt lookup on CNN/DailyMail summarization, versus up to 1.5x with a draft model on ShareGPT chat. It notes that prompt lookup 'works particularly well when the large model repeats parts of the prompt in its answers'.
2. Sampling that always picks the most likely token behaves the same way. Leviathan Tables 2-3 (Sec. 4, p.6-7, beyond the assigned sections) show alpha and speedup are higher at temperature 0 than at temperature 1. With T5-small as the draft on English-to-German translation: alpha 0.75 and 3.4X at temp 0, versus 0.62 and 2.6X at temp 1.
3. A worked example (my arithmetic from the paper's Eq. 1 and Theorem 3.8, with gamma = 5 guesses and c = 0.05):
   - alpha = 0.8 gives about 3.7 tokens per big-model pass, about 2.95x faster.
   - alpha = 0.5 gives about 2.0 tokens per pass, about 1.57x faster.
   - alpha = 0.2 gives about 1.25 tokens per pass, about 1.00x (no gain).

Lists and creative text, marked as inference: no assigned source mentions lists or creative writing by name. [Inference] Lists help because of repeated structure (bullets, numbering, parallel phrasing) and items echoed from the prompt; these produce many 'easy' tokens. [Inference] Creative writing usually has many reasonable next words and is often sampled at higher temperature. The big model's probability then spreads over many tokens, so the cheap guesser's pick often fails the check, which means lower alpha and fewer free tokens. The temperature result above supports this, but only indirectly.

On 2507.09019: it gives no comparison of acceptance by task type. It only says the benefit 'depends heavily on factors like the draft model's accuracy and the typical response length' (Sec. 4, p.9).

## Why the speedup shrinks with more users

Step 1: why there is a speedup at all (sourced). For one user, generating each token means reading all of the model's weights from GPU memory. The arithmetic units mostly sit idle waiting on that memory. Leviathan Sec. 1, p.1: inference is 'often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication', 'so additional computation resources might be available'. Gordic ('Scheduler'): decode is 'memory-bandwidth-bound, since we still need to load all LLM weights (and KV caches) just to compute one token'. The handbook: 'Much of their compute sits unused while waiting on memory access.' Speculative decoding spends that idle arithmetic on checking several guessed tokens in the same weight read. The paper's math builds this in as an assumption. Sec. 3.3, p.4 assumes 'we can run gamma + 1 concurrent evaluations of Mp in parallel without increasing the walltime'. In other words, checking 6 tokens is assumed to cost the same as producing 1.

Step 2: many users eat that slack (sourced, plus one connecting inference). A serving system batches many users together, so one weight read serves all of them. Gordic's roofline passage ('Benchmarks and auto-tuning - latency vs throughput'): 'below a saturation batch B_sat, the step time is dominated by HBM bandwidth', 'so step latency is nearly flat'. But 'Beyond B_sat, the kernels become compute-bound and step time grows roughly with B'. [Inference: Gordic does not apply this to speculation.] Once enough users share the GPU, the idle arithmetic that speculation relied on is already producing other users' tokens. The GPU has moved from memory-bandwidth-bound to compute-bound. The supplementary TurboSpec paper (Sec. 4.1.2) supports this: 'increasing the batch size tends to produce more tokens at the same cost'.

Step 3: in the compute-bound regime, extra checking work is no longer free (sourced). Speculation always does more total arithmetic. Leviathan Sec. 3.4, p.4: 'the total number of arithmetic operations might be higher than that of the standard decoding algorithm' (Theorem 3.11). My arithmetic from Theorem 3.11 with gamma = 5 and c_hat = 0.05: about 1.7x more arithmetic at alpha = 0.8, and about 3.2x at alpha = 0.5. Leviathan's own Discussion (Sec. 6, p.8, beyond the assigned sections): 'our method is not helpful for configurations where additional computation resources are not available.' [Inference] Under heavy load, the Theorem 3.8 assumption (gamma+1 checks at no extra time) stops holding, so the formula overstates the speedup.

Step 4: rejected guesses waste compute other users could have used (sourced). Every rejected guess was drafted and verified for nothing. Handbook, 'Don't ignore wasted compute': 'When the target model rejects too many draft tokens, your GPU still spends time generating and verifying them.' The handbook's DSpark section: that wasted verification 'under heavy load ... competes with batch capacity that could serve other requests.' TurboSpec Sec. 1 (supplementary): such waste 'may be acceptable under low system load, [but] they become prohibitive if the acceptance rate drops or the request rate to the system increases.' The draft model also takes GPU memory, which reduces how many requests fit in a batch (handbook, 'Watch out for memory overhead').

Step 5: what was measured.
- Handbook (BentoML test, Llama-3.3-70B, H100): with TP=1, throughput 'plateaued earlier (around 20-30 concurrent requests)', and gamma = 5 showed latency spikes at 40+ concurrent requests. The page's only stated reason is that draft-target 'coordination ... might bring overhead at higher loads'. The page never uses the words 'compute-bound' or 'batch size'.
- Supplementary vLLM 2024 post: at high request rates, 'The extra compute required to propose and verify tokens can sometimes slow down the system when it is already compute-bound'. It reports 1.4x slowdown (ShareGPT) and 1.8x slowdown (CNN/DailyMail) for Llama3-70B on 4xH100.
- TurboSpec Sec. 6.1: 'significant performance deterioration' at batch sizes of 16 or more.
- Mitigations from both supplementary sources: shorten the guess length when load is high, or switch speculation off.

Important caveat about 2507.09019 (Agrawal et al.): its case study (Sec. 4, p.8-9) does NOT test multiple load levels. It runs one low load (0.25 QPS, Llama-3 70B, 4xH100). Its lessons are about measurement:
- Tokens arrive in bursts, so the time-between-tokens distribution is 'highly bimodal'.
- Median TPOT is '1.3x lower', yet P99 TPOT is '1.16x higher'.
- TTFT is consistently higher because of the draft model's prefill.
- So reporting only averages or normalized numbers can mislead (Anti-Patterns 7 and 8).

[Inference] For a beginner the takeaway is that a single low-load benchmark like this one cannot show how speculation behaves with many users. Test across several load levels and look at tail latency, not just averages.
