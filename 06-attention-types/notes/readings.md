# Readings: attention designs in modern LLMs (Task 6 notes)

Plain-language notes for the slides. Every number below comes from a source I opened (listed at the end), with the table/figure/section when I could see it. Anything I could not check is marked **not verified**. "Inference" labels my own reasoning, not a source claim.

Heads-up on how sources were read: web pages were read through a fetch tool that summarises the page, so figure values that exist only as plots are not available (the fetch said so each time). Treat table values as "as reported by the fetch of the paper"; double-check against the PDF before putting a number on a slide.

---

## 0. Glossary and the baseline

- **Token**: a chunk of text (word piece) the model reads.
- **Q, K, V (query, key, value)**: three vectors per token. A query asks "what am I looking for?", keys say "what do I contain?", values are "what I give back if you attend to me".
- **Head**: one independent attention calculation. A layer runs many heads in parallel so they can specialise (Raschka: local, semantic, positional patterns).
- **KV cache**: when generating text one token at a time, the K and V of all earlier tokens are stored so they are not recomputed. It is fast but eats memory.
- **Sequence length N (or L, T)**: number of tokens in the context.
- **Causal mask**: a token may only look at earlier tokens (Raschka: the upper-right triangle of the T x T matrix is excluded).
- **HBM / SRAM**: the GPU's big slow memory vs. its tiny very fast on-chip memory (FlashAttention paper, sec 2.1).

### Baseline: full multi-head attention (MHA)

For each head: `Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V`, where d is the head size. Steps (Raschka): project input X with Wq, Wk, Wv; QK^T gives token-to-token relevance scores; softmax turns them into the attention matrix A; A applied to V gives the output Z.

Costs (standard facts; the sources state them in these words):
- **Compute grows with the square of sequence length**: the score matrix is N x N. Welch Labs (via a summary of the video): "pattern size scales with the square of the input token count". FlashAttention paper: standard attention needs "time and memory quadratic in sequence length".
- **KV cache grows linearly with length**: one more token adds one more K and V per head per layer.
- **KV-cache bytes per token = 2 x layers x KV heads x head size x bytes per number** (the 2 is for K and V; bytes is 2 for fp16/bf16). This is the formula in the Welch Labs video (2 x tokens x head dimension x heads x layers in fp16) and DeepSeek-V2 Table 1 (in elements, before multiplying by bytes).
- Worked example from the Welch Labs video page for DeepSeek-V3/R1 (61 layers, 128 heads, head dim 128): MHA = 3,997,696 bytes per token (about 4 MB). My check: 2 x 61 x 128 x 128 x 2 bytes = 3,997,696. It matches. At 100,000 tokens the video says roughly 400 GB.

---

## 1. MHA (multi-head attention)

- **Problem it solves**: one attention pattern is too limited; different heads can look for different things.
- **Core idea**: every head has its own Q, K and V projections. KV heads = query heads.
- **Memory/compute**: KV cache per token = 2 x layers x n_h x d_h x bytes (DeepSeek-V2 Table 1: 2 n_h d_h l elements). Compute is O(N^2) per layer.
- **Examples named by Raschka**: GPT-2, OLMo 2 7B, OLMo 3 7B.
- **Limits**: the KV cache is the biggest of all variants. Reading the whole cache each decoding step makes generation memory-bandwidth bound (GQA paper, sec on memory bandwidth).
- **Improves on**: it is the starting point.

---

## 2. MQA and GQA (share keys/values across heads)

**Problem**: during generation the model must load all keys and values from memory at every step; that bandwidth is the bottleneck (GQA paper).

**Core idea**
- **MQA (multi-query)**: all query heads share ONE K head and ONE V head. Cuts the data loaded by a factor of H (GQA paper). Cheapest, but quality can drop.
- **GQA (grouped-query)**: query heads are split into G groups; each group shares one K/V head. G = 1 is MQA, G = number of heads is MHA. Raschka calls the middle "the sweet spot".
- Formulas (DeepSeek-V2 Table 1, elements per token): MQA = 2 d_h l; GQA = 2 n_g d_h l (n_g = groups); MHA = 2 n_h d_h l.
- Shape example (Welch Labs page): Llama 3 uses groups of eight, about 8x smaller cache than MHA.

**GQA paper (Ainslie et al., arXiv 2305.13245) results**
- Convert an existing MHA checkpoint by **mean-pooling** the K and V projection matrices of the heads in each group, then "uptrain" for a short while. Uptraining used about 5% of the original pre-training compute (600 TPUv3 chip-days for alpha = 0.05). Mean pooling beat picking the first head, which beat random init (Fig 4; exact values not given in the text).
- **Table 1** (T5, summarisation + translation + QA; time per sample per TPUv4 chip): 

| Model | Time (s) | Avg R1 |
|---|---|---|
| MHA-Large | 0.37 | 46.0 |
| MHA-XXL | 1.51 | 47.2 |
| MQA-XXL (5% uptrained) | 0.24 | 46.6 |
| GQA-8-XXL (5% uptrained) | 0.28 | 47.1 |

  So GQA-8 keeps quality within 0.1 average ROUGE-1 of MHA-XXL while being about 5.4x faster (1.51 vs 0.28 s; my division). MQA is slightly faster but loses 0.6.
- Fig 6: 8 groups chosen as a middle ground; going from 1 group to 8 adds modest cost.
- Stability (Appendix A): MQA trained from scratch had loss spikes and diverged on long-input fine-tuning; uptrained GQA looked stable.

**Limits (authors)**: only encoder-decoder T5 tested; ROUGE is a flawed metric; no comparison with a GQA model trained from scratch; cache savings matter most for long generation.
**Raschka's view**: GQA is still the robust, easy-to-train default: Llama 3 8B, Qwen3 4B, Gemma 3 27B, Mistral Small 3.1 24B, SmolLM3 3B, Llama 4 Maverick, Qwen3 235B-A22B, Step 3.5 Flash 196B, Sarvam 30B; MiniMax M2.5 and Nanbeige 4.1 per his "Spring Architectures" article (not opened).
**Improves on MHA**: smaller cache and less memory traffic with little quality loss. It does NOT change the O(N^2) compute.

---

## 3. MLA (multi-head latent attention, DeepSeek)

**Problem**: GQA/MQA shrink the cache by removing heads, which costs quality. Can we shrink the cache without removing heads?

**Core idea (DeepSeek-V2, sec 2.1)**: compress, don't share.
- Cache one small latent vector per token: `c_t = W_DKV h_t` (W_DKV is d_c x d). Rebuild per-head keys and values on demand: `k = W_UK c_t`, `v = W_UV c_t` (Eq 9 to 11).
- Queries are also compressed (d'_c), but that only saves training activation memory, not KV cache (Eq 12-13).
- **Decoupled RoPE**: rotary position encoding can't be folded into the up-projection, so a small extra key k^R (d_h^R dims, shared across heads) carries position and is cached too (Eq 14-17). Cache per token = (d_c + d_h^R) x layers.
- **Weight absorption**: W_UK can be folded into the query side and W_UV into the output projection, so the full keys/values are never materialised at inference.
- DeepSeek-V2 sizes (sec 3.1.2): n_h = 128, d_h = 128, d_c = 512 (= 4 d_h), d'_c = 1536, d_h^R = 64 (= d_h/2), hidden 5120, 60 layers.

**Table 1 (DeepSeek-V2), KV elements per token**

| Mechanism | Cache | Capability label |
|---|---|---|
| MHA | 2 n_h d_h l | Strong |
| GQA | 2 n_g d_h l | Moderate |
| MQA | 2 d_h l | Weak |
| MLA | (d_c + d_h^R) l ~ 4.5 d_h l | Stronger |

The paper says MLA's cache equals GQA with only 2.25 groups "but its performance is stronger than MHA".

**Headline results**
- DeepSeek-V2 vs DeepSeek 67B (abstract): 236B total / 21B active params, 128K context, **93.3% smaller KV cache**, **42.5% lower training cost**, **up to 5.76x max generation throughput**, pretrained on 8.1T tokens. Section 3.2.3: over 50K tokens/s on one 8xH800 node.
- The MHA/GQA/MQA vs MLA ablation tables (Appendix D, Tables 8 and 9) were not in the text I could read: **not verified** (I only have the sentence that MLA beats MHA). Raschka's article states the same finding in words: GQA worse than MHA, MLA competitive or slightly better when tuned.
- Welch Labs (video page/description only, no transcript): for V3/R1, MLA cache = 576 x 61 x 2 = 70,272 bytes per token vs 3,997,696 for MHA, about 56.9x smaller. Author's own estimate for V2 (30 layers, 32 heads): 491,520 vs 34,560 bytes per token, about 92.8%; he notes the paper gives 93.3% without a stated method. A second summary of the video (lilys.ai) lists about 4 MB (MHA), about 500 KB (GQA-8), about 70 KB (MLA) per token for R1.

**Limits/costs**: more complex to implement and serve (Raschka). Extra decoupled-RoPE key. Raschka passes on colleagues' view that MLA works well mainly at larger sizes; under about 100B parameters GQA seems easier (hearsay, not a measurement).
**Examples (Raschka)**: DeepSeek V3, Kimi K2, GLM-5, Ling 2.5, Mistral Large 3, Sarvam 105B.
**Improves on GQA**: much smaller cache AND (claimed) no quality loss. Compute is still O(N^2); only memory improves.

---

## 4. Sliding-window (local) attention

**Problem**: full attention reads every earlier token. Most useful context is nearby.
**Core idea**: each token attends only to the last W tokens (Raschka). Cost per token becomes O(W) instead of O(N); the cache for those layers can be capped at W tokens. Models mix local layers with occasional global layers so information can still travel across the whole text.
**Numbers (Raschka)**: Gemma 2 = 1:1 local:global, window 4096. Gemma 3 = 5:1, window 1024 (Gemma 3 ablation: perplexity hurt only slightly). OLMo 3 and Arcee Trinity = 3:1. Xiaomi MiMo-V2-Flash = 5:1 with window 128. Often combined with GQA (Gemma 3).
**Examples (Raschka)**: Gemma 3 27B, OLMo 3 32B, Xiaomi MiMo-V2-Flash, Arcee Trinity, Step 3.5 Flash, Tiny Aya.
**Limits**: the window is hard-coded; distant facts are only reachable through the few global layers. MiniMax tried converting M2 to a sliding-window hybrid by continued pretraining: performance dropped as context grew, which they called unacceptable for agent use (no numbers published). Their guess: retrieval/induction heads form early in pretraining and are hard to change later; they say it is not an attention-sink issue.
**Improves on full attention**: cache and compute per token stop growing with length in local layers.

---

## 5. Sparse attention (learned selection): NSA, DSA, CSA/HCA

**Problem**: a fixed window is blunt. Can the model choose which earlier tokens matter, and do it fast and trainably?

### 5a. NSA, Native Sparse Attention (Yuan et al., arXiv 2502.11089)
- **Idea**: three parallel branches mixed by learned gates: (1) **compression** of blocks into summary tokens (block length l = 32, stride d = 16), (2) **selection** of the top blocks (block size 64, n = 16 selected, including 1 initial and 2 local blocks), (3) a **sliding window** (w = 512).
- "Native" = trained sparse from the start, with GPU-friendly block kernels; query heads in a GQA group load the same selected KV blocks.
- **Results**: 27B-parameter model, 3B active, about 260B pretraining tokens (the paper also says 270B in sec 4.1: internal inconsistency). Table 1: average 0.456 vs 0.443 for full attention, better on 7 of 9 metrics. Table 2 (LongBench): 0.469 vs 0.437 full vs 0.423 Exact-Top. Table 3: AIME 24 after fine-tuning: 0.121 (8k) and 0.146 (16k) vs 0.046 and 0.092 for full attention. 
- **Speed at 64k context** (Figs 1 and 6): decoding up to 11.6x, forward 9.0x, backward 6.0x (FlashAttention-2 baseline for training). Table 4: at 65,536 context NSA reads 5,632 tokens vs 65,536 for full.
- **Limits**: three branches plus custom kernels; not a drop-in for existing checkpoints; no dedicated limitations section.

### 5b. DSA, DeepSeek Sparse Attention (DeepSeek-V3.2, arXiv 2512.02556)
- **Idea**: a **lightning indexer** gives each earlier token a score I_{t,s} = weighted sum over indexer heads of ReLU(q . k) (cheap, can run in FP8). A **top-k selector** keeps k = 2048 entries; real attention (MLA in MQA mode) runs only on them.
- **Complexity**: core attention O(L^2) becomes O(L k). The indexer itself is still O(L^2) but much cheaper than the main attention.
- **Convertible**: continued pretraining from DeepSeek-V3.1-Terminus (128K context): dense warm-up 1,000 steps, 2.1B tokens (indexer only, lr 1e-3); sparse stage 15,000 steps, 943.7B tokens (lr 7.3e-6).
- **Results**: "no substantial degradation" on short and long-context benchmarks; AA-LCR 4 points higher than V3.1-Terminus in reasoning mode. Cost per million tokens at 128K is plotted in Fig 3 (H800 at $2/GPU-hour; axes span roughly $0.1 to $0.7 prefill and $0.4 to $2.4 decode); exact values not in the text I got.
- **Authors' limits**: less world knowledge than frontier proprietary models (fewer training FLOPs), weaker token efficiency, trails frontier on complex tasks (these are model-level limits, not DSA-specific).
- **Examples (Raschka)**: DeepSeek V3.2, GLM-5 (MLA + DSA).
- **Improves on NSA/SWA**: learned, content-based selection, and it can be bolted on to an existing model by short continued training.

### 5c. DeepSeek-V4: CSA and HCA (arXiv 2606.19348, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence")
- I found and opened the report (it exists). The fetch could not show its Conclusion/Limitations section (listed on p. 44): limits **not verified**.
- **CSA (Compressed Sparse Attention, sec 2.3.1)**: compress every m = 4 tokens' KV into one entry, then use DSA-style top-k selection over the compressed entries. k = 512 (V4-Flash), 1024 (V4-Pro) (sec 4.2.1); the indexer runs in FP4.
- **HCA (Heavily Compressed Attention, sec 2.3.2)**: compress by m' = 128, dense attention over those few entries, no selection.
- **Both** add a sliding window of 128 uncompressed recent tokens (sec 2.3.3), to keep local detail. Layers alternate CSA and HCA (Flash: first two layers pure sliding window; Pro: first two HCA).
- **Efficiency at 1M context vs DeepSeek-V3.2 (sec 1, Fig 1, sec 2.3.4)**: V4-Pro about 27% of single-token FLOPs and about 10% of KV cache; V4-Flash about 10% of FLOPs and about 7% of KV cache. About 2% of the cache of a BF16 GQA8 baseline (head dim 128). Fig 1 labels "3.7x / 9.8x lower" (FLOPs) and "9.5x / 13.7x smaller" (KV) could not be matched to models: **not verified** mapping.
- **Sizes (sec 4.2.1)**: Pro 1.6T total / 49B active, 61 layers; Flash 284B total / 13B active, 43 layers.
- **Why it matters here**: it combines three ideas from this deck: compression (MLA-like), learned selection (DSA), local window (SWA).

---

## 6. Linear attention and hybrids: Gated DeltaNet, Kimi Linear

**Problem**: even with tricks, full attention is O(N^2) compute and a cache growing with N. Raschka names 128k, 256k, 1M-token contexts as where this hurts.

**Core idea**: replace the N x N attention with a small fixed-size memory (a matrix "state") that is updated token by token. Memory stays constant with length and compute is linear.
- **Gated DeltaNet** (Qwen3-Next, Qwen3.5; Raschka): per token compute q, k, v plus two learned gates (alpha, beta); write to a small fast-weight memory with a **delta rule** (it corrects what the memory already predicts instead of just adding). Related to Mamba-2. Used in a 3:1 pattern (3 Gated DeltaNet blocks : 1 gated full-attention block). The "gated attention" block adds an output gate, zero-centered QK-Norm, and partial RoPE.
- **Kimi Linear (arXiv 2510.26692)**: **Kimi Delta Attention (KDA)** refines Gated DeltaNet by using a **channel-wise** forget gate (each feature has its own decay) instead of one scalar per head. State update: diagonal decay, then a delta-rule correction, then write the new key-value pair. Hybrid is 3 KDA layers : 1 full MLA layer; the MLA layers use no positional encoding (NoPE).
- Other hybrids named by Raschka: Ling 2.5 (Lightning Attention + MLA), Nemotron 3 Nano/Super (Mamba-2 + sparse MoE, few attention layers).

**Kimi Linear results**
- 48B total / 3B active parameters (256 experts, 8 active + 1 shared). Architecture comparison on 1.4T tokens; released checkpoint trained on 5.7T tokens, up to 1M context.
- Ablation: 3:1 gives lowest validation perplexity 5.65 vs 5.77 for pure full attention.
- 1.4T-token comparison: MMLU-Pro 51.0 (MLA 47.2, GDN-H 47.9); RULER at 128k 84.3 (MLA 81.3, GDN-H 80.5); long-context average 54.5 (MLA 52.2).
- After SFT: GPQA-Diamond 62.1 vs 57.1 (MLA); AIME 2025 21.3 vs 20.6.
- Efficiency: "KV cache usage by up to 75%"; abstract: up to 6x decoding throughput at 1M context (Fig 1b shows 6.3x TPOT vs MLA, but that gain comes partly from freed memory allowing bigger batches; at batch size 1 Fig 7b shows about 2.3x at 1M tokens). About 1.16x compute efficiency vs MLA under compute-optimal training.
- Synthetic tasks (Palindrome, MQAR, Stack, lengths 256 to 2,048): KDA best; Mamba2 fails all three.

**Limits**: the authors say pure linear attention still struggles with exact copying and fine-grained retrieval, hence the full-attention layer every fourth block. Trails on some benchmarks (EvalPlus 60.2 vs 63.1 for GDN-H; LongBench v2 35.0 vs 36.1 for MLA). Hardware/inference stacks less mature. Raschka: hybrids are mainly a long-context efficiency story, not a raw quality story; he gets better tokens/sec locally with GPT-OSS (GQA).
**Improves on full attention**: constant-size memory and linear compute for most layers.

---

## 7. FlashAttention (Dao et al., arXiv 2205.14135)

**Problem**: attention is slow mostly because of memory traffic, not arithmetic. Standard attention writes the N x N score matrix S and probabilities P to slow HBM and reads them back (Algorithm 0).
**Core idea**: an exact algorithm (same output, no approximation). **Tile** Q, K, V into blocks that fit in fast SRAM; compute softmax **online** with running statistics; fuse everything into one kernel; in backward pass **recompute** attention instead of storing the N x N matrix.
- Online softmax (sec 3.1): m(x) = max(m(x1), m(x2)); l(x) = e^(m(x1)-m(x)) l(x1) + e^(m(x2)-m(x)) l(x2).
- Hardware (A100, sec 2.1): HBM 40-80 GB at 1.5-2.0 TB/s; SRAM 192 KB per SM x 108 SMs at about 19 TB/s.
- **Theorem 2**: standard attention needs Theta(Nd + N^2) HBM accesses; FlashAttention needs Theta(N^2 d^2 / M) (M = SRAM size). Memory use becomes linear in N.
- **Figure 2 (GPT-2 medium, N = 1024, A100)**: standard vs Flash: GFLOPs 66.6 vs 75.2 (more math!), HBM read/write 40.3 GB vs 4.4 GB, runtime 41.7 ms vs 7.3 ms. This is the best slide: more arithmetic, still 5.7x faster (my division), because memory traffic dominates.
- **Training speedups** (abstract): 15% on BERT-large (seq 512) vs MLPerf 1.1 record (the fetch of the paper also gave 17.4 vs 20.0 min); 3x on GPT-2 (seq 1K); 2.4x on Long-Range Arena (1K-4K). Table 2: GPT-2 small 9.5 days (HuggingFace) vs 4.7 (Megatron) vs 2.7 (Flash), same perplexity 18.2 (the table row says 3.5x, the text says "up to 3x": paper inconsistency); GPT-2 medium 21.0 vs 11.5 vs 6.9 days.
- **Quality from longer context**: GPT-2 with 4K context got 0.7 better perplexity than 1K baseline while training 30% faster; Path-X 61.4%, block-sparse Path-256 63.1% (first Transformers above chance). Block-sparse FlashAttention: 2.8x on LRA.
- **Limits**: each new attention variant needs a hand-written CUDA kernel; multi-GPU IO not analysed. It does NOT cut the number of FLOPs: compute is still O(N^2); it fixes memory traffic and the N x N memory.
- **How it fits**: it is orthogonal to the others. It speeds up whichever attention you use (NSA compares to FlashAttention-2; sparse and window kernels build on the same tiling idea).

---

## 8. Comparison table

KV cache per token in elements (multiply by bytes). l = layers, h = KV heads, d = head size, W = window, k = tokens kept.

| Family | KV cache per token | Compute scaling | From scratch / convertible | Example models (sources name them) |
|---|---|---|---|---|
| MHA | 2 l n_h d | O(N^2) | scratch (baseline) | GPT-2, OLMo 2 7B, OLMo 3 7B (Raschka) |
| MQA | 2 l d | O(N^2) | convertible (uptrain ~5% compute, GQA paper) | none named in the sources I read |
| GQA | 2 l n_g d | O(N^2) | convertible from MHA by mean-pooling + ~5% uptraining | Llama 3 8B, Qwen3 4B, Gemma 3 27B, Mistral Small 3.1, Llama 4 Maverick, Qwen3 235B-A22B, Sarvam 30B (Raschka) |
| MLA | (d_c + d_h^R) l ~ 4.5 d_h l in DeepSeek-V2 | O(N^2) | trained in (no conversion recipe in my sources) | DeepSeek V3, Kimi K2, GLM-5, Ling 2.5, Mistral Large 3, Sarvam 105B (Raschka) |
| Sliding window | capped at W tokens in local layers (my reading of Raschka) | O(N W) in local layers | trained in; MiniMax's CPT conversion failed | Gemma 2/3, OLMo 3, MiMo-V2-Flash, Arcee Trinity, Step 3.5 Flash |
| NSA | compressed + selected + window (no per-token formula given); reads 5,632 of 65,536 tokens at 64k decoding (Table 4) | sub-quadratic, 9.0x fwd at 64k | trained natively sparse | none named |
| DSA | MLA latent cache; attends top k = 2048 | O(L k) + cheap O(L^2) indexer | convertible via continued pretraining | DeepSeek V3.2, GLM-5 |
| CSA/HCA | compressed 4x (CSA) / 128x (HCA) + window 128; ~7-10% of V3.2 cache at 1M | CSA top-k 512/1024 | trained in | DeepSeek-V4 Flash/Pro |
| Linear (GDN / KDA) | fixed-size state, independent of length | O(N) | trained in (hybrid with 1 full layer in 4) | Qwen3-Next, Qwen3.5 (GDN); Kimi Linear (KDA); Ling 2.5 (Lightning) |
| FlashAttention | does not change cache | still O(N^2) FLOPs, memory O(N) | any model, kernel only | used as baseline everywhere |

---

## 9. What the sources say about full attention staying useful (MiniMax M2)

From "Why Did MiniMax M2 End Up as a Full Attention Model?" (they say efficient attention is not yet reliably better in production):
1. **Evaluation is the bottleneck.** A small-scale Lightning-Attention/full hybrid matched full attention on MMLU, BBH, MATH, LongBench; at larger scale it was clearly weaker on complex multi-hop reasoning. Their proxy metrics may not hold at scale.
2. **Hard tasks are costly to measure**, and some problems only show up at scale (they cite precision problems from M1 RL training).
3. **Too many moving variables** (data mix, optimizer); a conclusion from last month's data may not hold.
4. **Infrastructure is immature**: many linear architectures are memory-bound even in training; open problems: low-precision state storage, prefix caching, speculative decoding with a linear backbone. Linear attention only wins on paper beyond "a few thousand tokens".
5. **Compute framing**: efficient attention saves compute; for equal quality they would bet on other approaches under scaling laws.
6. **SWA conversion failed** (see section 4); global patterns like retrieval and induction heads form early and are hard to change later.
No benchmark scores are published in the piece. Raschka (separate source) says similar: no public same-data comparisons exist, hybrids are mainly a long-context efficiency play.

---

## 10. Tie-in to your Qwen3-0.6B measurement (inference, not a source claim)

Your facts (4,000-token WikiText documents): median head needs only 34 of about 3,584 earlier tokens for 90% of attention; first token is an attention sink from layer 3 (20 to 75% of attention); layers 0 to 2 local; only 4 of 448 heads need over 1,024 tokens; first token + latest 128 tokens hold 62 to 95% of attention per layer.

**Supports (inference)**
- **Sliding window + sink tokens**: layers 0-2 being local and most heads needing a few dozen tokens fit the SWA idea (Gemma 3 uses 1024; Xiaomi 128). The sink fact suggests keeping the first token plus a recent window; this resembles NSA's design (initial block + local blocks + window of 512) and CSA/HCA's 128-token window. Note: sink-keeping is not itself described in the sources I read beyond NSA's "initial block" and V4's "attention sink logits".
- **Learned sparse selection (DSA, NSA, CSA)**: if 34 tokens cover 90%, a top-k of 2048 (DSA) or 512/1024 (CSA) would be generous at 4K, so savings would show up only at much longer lengths.
- **Hybrid with few global layers**: only 4 of 448 heads need over 1,024 tokens, which suggests a few global heads/layers could be enough, in the spirit of Gemma's 5:1 or Kimi's 3:1 ratio.

**Argues against / cautions (inference)**
- MiniMax's warning: local patterns in a 4K WikiText sample may not reflect retrieval, multi-hop reasoning, or agent tasks; the few heads needing long range could be the ones that matter. Your test text is plain prose, 4K tokens, a small model.
- Linear attention: a measurement of where attention goes does not show whether a fixed-size memory can reproduce exact retrieval, which is where Kimi Linear's authors say pure linear struggles.
- MLA/GQA save cache, not compute, so these measurements neither support nor refute them; the cache formula applies as-is (Qwen3-0.6B config values are not in my sources: **not verified**, read them from the model config).
- Cutting down to 34 tokens at 90% leaves 10% of attention mass elsewhere; the sources do not say how much quality loss that causes.

---

## Sources actually read

1. Raschka, A Visual Guide to Attention Variants in Modern LLMs (22 March 2026): https://magazine.sebastianraschka.com/p/visual-attention-variants . Read via fetch summary; the article has no formulas and its KV-cache figures (11, 12, 14, 25) are plots with no stated values.
2. GQA: https://arxiv.org/abs/2305.13245 and https://arxiv.org/pdf/2305.13245
3. DeepSeek-V2: https://arxiv.org/abs/2405.04434 , https://arxiv.org/pdf/2405.04434 , https://arxiv.org/html/2405.04434 . Appendix D ablation tables (8 and 9) were NOT in what I could read.
4. Welch Labs video "How DeepSeek Rewrote the Transformer [MLA]" (published 5 March 2025, 18:09): I could not open YouTube (robots-blocked). I used a mirror page with the title/description/technical notes, https://yt.chop.dev/en/v/0VLAoVGf_74 (no transcript), and a third-party notes summary, https://lilys.ai/ja/notes/deepseek-ai-20260114/deepseek-transformer-rewrite . The video itself was not watched; the summary lacks the RoPE explanation.
5. FlashAttention: https://arxiv.org/abs/2205.14135 and https://arxiv.org/pdf/2205.14135
6. NSA: https://arxiv.org/pdf/2502.11089
7. DeepSeek-V3.2: https://arxiv.org/pdf/2512.02556 ; DeepSeek-V4: https://arxiv.org/pdf/2606.19348 (found by search and opened; its limitations section was not visible)
8. Kimi Linear: https://arxiv.org/pdf/2510.26692
9. MiniMax: https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model (the same piece is also at https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model, found in search, not opened)

Found in search but not opened: Raschka's MiniMax M2 article (https://www.sebastianraschka.com/blog/2026/minimax-m2-technical-report.html), DeepSeek-V3.2-Exp coverage, V4 coverage by third parties.
