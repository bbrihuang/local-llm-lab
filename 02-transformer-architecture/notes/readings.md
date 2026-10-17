# Readings: how a transformer works

Notes for Task 2 (a ~20-minute talk on transformer architecture). Written for a beginner.

**Number rule:** every number below comes from one of the sources I read (named next to it). The Qwen3-1.7B numbers (28 layers, hidden size 2,048, 16 query heads, 8 KV heads, head size 128, 40,960 default context) come **from your own model config, not from these five sources**. The KV-cache sizes are worked out from them. Anything I could not check is marked **not verified**.

How I read the two videos:
- **3Blue1Brown:** I used the written lesson at 3blue1brown.com/lessons/attention ("Deep Learning Chapter 6", text adaptation by Justin Sun). It has the same title as the video. I did not watch the video or read a transcript.
- **Karpathy:** I used the YouTube description and chapter list, the `ng-video-lecture` repo (README and `gpt.py`), the nanoGPT README, and a third-party AI summary of the video with timestamps (videohighlight.com). I did not read the full transcript, so I have fewer loss numbers than the video gives.

---

## 1. Glossary

| Term | Plain meaning |
|---|---|
| **Token** | A piece of text the model works with: a word, part of a word, or a single character. Karpathy's tiny model uses single characters. Real models use sub-word pieces (BPE). |
| **Embedding** | A list of numbers (a vector) that stands for a token. Each token ID looks up one row of a big learned table. |
| **Attention** | The step where each token looks at the other tokens and pulls in information from the ones that matter to it. |
| **Head** | One independent copy of attention, with its own query/key/value weights. Several heads run side by side and their outputs are combined. |
| **MLP / feed-forward** | A small two-layer neural network applied to each token on its own: widen, apply a non-linearity, narrow back. |
| **Residual connection** | "Add the input back to the output": `x = x + layer(x)`. Each layer only adds an update. It keeps training stable in deep stacks. |
| **LayerNorm / RMSNorm** | Rescales each token's vector so its values stay in a sane range. RMSNorm is a simpler, cheaper version used by most modern models. |
| **Logits** | The raw scores the model gives every token in the vocabulary for "what comes next". |
| **Softmax** | Turns a list of scores into probabilities: all positive, summing to 1. |
| **Sampling / temperature** | Picking the next token at random according to the probabilities. Temperature divides the logits first: low means safer and more predictable, high means more random. |
| **KV cache** | Saved keys and values of earlier tokens, so each new token does not recompute them. |
| **Positional encoding / RoPE** | Attention on its own does not know word order, so position information is added. The 2017 model added sine/cosine vectors. Modern models rotate queries and keys by an angle that depends on position (RoPE, "rotary"). |
| **Decoder-only** | A model with only the "decoder" half: a single stack that reads the text so far and predicts the next token, with a causal mask. GPT, Llama, Qwen and the other models below are all decoder-only. |

---

## 2. Following one prompt through the model

Take the prompt "The cat sat on the" in a decoder-only model like Qwen3-1.7B.

**Step 1: text to tokens.** A tokenizer splits the text into pieces and maps each piece to an integer ID. Karpathy's code does this at character level: `chars = sorted(list(set(text)))`, with `stoi` and `itos` tables for the mapping. The original Transformer used BPE with about 37,000 shared tokens for English-German (Annotated Transformer).

**Step 2: tokens to embeddings.** Each ID looks up a learned vector of size `d_model`: 512 in the 2017 model (Alammar, Annotated), 384 in Karpathy's tiny model, 12,288 in GPT-3 (3Blue1Brown), and 2,048 in Qwen3-1.7B (your config). Karpathy adds a learned position embedding: `x = tok_emb + pos_emb`. The 2017 model adds sine/cosine positions:
`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`, `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))` (Annotated).

**Step 3: N transformer blocks.** The same kind of block repeats N times: 6 in the 2017 model and in Karpathy's model, 96 in GPT-3, 28 in Qwen3-1.7B. Each block (pre-norm style, as in Karpathy's `gpt.py`):

```
x = x + Attention(Norm(x))   # tokens talk to each other ("communication")
x = x + MLP(Norm(x))         # each token thinks on its own ("computation")
```

*Attention.* Each token's vector is multiplied by three learned matrices to get a query **Q**, a key **K** and a value **V**:

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

- `Q K^T`: every query is compared with every key using a dot product, so "how relevant is token j to token i".
- `/ sqrt(d_k)`: keeps scores from growing too large, which would make softmax gradients vanish. With d_k = 64 you divide by 8 (Alammar).
- **Causal mask:** before the softmax, scores for future positions are set to -infinity (Karpathy: `masked_fill(tril == 0, float('-inf'))`; the Annotated code uses -1e9). After the softmax they become 0, so a token can only look backwards.
- `softmax(...)`: each row becomes weights that sum to 1.
- `... V`: a weighted average of the value vectors is the update for that token.

*Multi-head attention.* Run h heads in parallel, each with its own W_Q, W_K, W_V, then concatenate and multiply by W_O:

    MultiHead(x) = Concat(head_1, ..., head_h) W_O

Example settings: 8 heads of size 64 (2017), 6 heads of size 64 (Karpathy), 96 heads with 128-dimensional queries and keys (GPT-3, per 3Blue1Brown). Qwen3-1.7B: 16 query heads of size 128 that share 8 key/value heads (GQA, Section 5).

*MLP (feed-forward).* Applied to each token separately:

    MLP(x) = W_2 · act(W_1 · x)

It widens and then narrows: 512 to 2,048 to 512 in 2017 (Annotated: d_ff = 2048), and `n_embd -> 4*n_embd -> n_embd` with ReLU in Karpathy's code. Modern models use SwiGLU instead of ReLU/GELU (Raschka).

**Step 4: final norm and logits.** After the last block, a final norm, then a linear layer (`lm_head`) turns the last token's vector into one score per vocabulary entry. Alammar's example uses a 10,000-word vocabulary, so 10,000 logits.

**Step 5: probabilities and sampling.**

    p_i = exp(z_i / T) / sum_j exp(z_j / T)     (T = temperature)

T = 1 leaves the logits unchanged. T < 1 sharpens the distribution toward the top token. T > 1 flattens it. Then you pick a token:
- **Greedy:** always take the highest-probability token (Alammar; `greedy_decode` in Annotated).
- **Sampling:** draw at random by probability (Karpathy: `torch.multinomial(probs, num_samples=1)`).
- **Beam search:** keep the best few partial outputs (Alammar's example: beam_size = 2).

The new token is appended and the loop repeats from Step 1 for that one token. This repeating loop is where the KV cache helps (Q5).

Note: none of the five sources writes out the temperature formula. It is the standard definition (**not verified against these sources**). Karpathy's `gpt.py` samples with no temperature, which is the same as T = 1.

---

## 3. Source by source

### 3.1 Jay Alammar, "The Illustrated Transformer" (2018)

**What it covers:** the original 2017 encoder-decoder Transformer, explained almost entirely with pictures, using a French-to-English example ("je suis étudiant" to "i am a student"). Written June 27, 2018.

**Most useful ideas and pictures for slides:**
1. **Zoom-in sequence:** black box, then encoder stack plus decoder stack, then one encoder (self-attention + feed-forward). A good opening for the talk.
2. **Self-attention in 6 steps** for one word: make q/k/v, dot-product scores, divide by 8 (sqrt 64), softmax, weight the values, sum. Then the same thing as matrices.
3. **"it" points to "The animal":** the clearest picture of why attention exists. Also two heads attending to different words ("animal" vs "tired").
4. **Multi-head:** 8 sets of Z matrices, concatenated and multiplied by W_O.
5. **Output end:** decoder vector, linear layer, 10,000 logits, softmax, pick a word. Greedy vs beam search.

**Numbers:** embedding size 512; q/k/v vectors size 64; 6 encoders + 6 decoders; 8 heads; divide scores by 8 = sqrt(64); example output vocabulary 10,000 words; a realistic vocabulary is "30,000 or 50,000"; toy training vocabulary 6 words; positional encoding values between -1 and 1; heatmap of 20 positions x 512 dimensions; beam_size = 2.

**Caveat:** it describes the 2017 *encoder-decoder* model with post-norm (`LayerNorm(x + sublayer(x))`). Qwen3 is decoder-only and pre-norm.

### 3.2 3Blue1Brown, "Attention in transformers, step-by-step" (Deep Learning Chapter 6)

**Used:** the written lesson on 3blue1brown.com. I did not watch the video.

**What it covers:** what attention *does to meaning*, with GPT-3 as the worked example and a careful parameter count.

**Most useful ideas and pictures for slides:**
1. **The same word, different meanings:** "American shrew **mole**", "one **mole** of carbon dioxide", "take a biopsy of the **mole**". The starting embedding is the same each time. Attention moves it toward the right meaning.
2. **"A fluffy blue creature roamed the verdant forest":** the query asks "are there adjectives in front of me?", the keys of "fluffy" and "blue" answer, and their values update "creature". Very slide-friendly.
3. **Attention pattern as a grid:** query-key dot products, then column-wise softmax. Masking sets future entries to -infinity. The grid grows with the **square of context length**, which is why context is expensive.
4. **Value matrix as "value-down" then "value-up":** a low-rank map (12,288 to 128 to 12,288) instead of a huge square matrix.
5. **Multi-head = many proposed changes added together:** each head proposes an update to each embedding, and all of them are summed into the original vector.

**Numbers (GPT-3):** embedding dimension 12,288; query/key space 128; each query or key matrix has 1,572,864 parameters; a naive full value matrix would be 150,994,944; about 6.3 million parameters per head; 96 heads per block (about 600 million parameters); 96 layers; attention totals "just under 58 billion" parameters, about a third of the model.

### 3.3 Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (Jan 17, 2023)

**Used:** the YouTube description and chapters, the `ng-video-lecture` README and `gpt.py`, the nanoGPT README, and a third-party timestamped summary (videohighlight.com). No full transcript.

**What it covers (first half, by chapter):** load tiny Shakespeare (about 1 million characters); character tokenizer; 90/10 train/val split; batches of chunks; a bigram baseline (loss about 2.5); then self-attention built up in four versions: (1) averaging past tokens with loops, (2) the same thing as a matrix multiply with a lower-triangular (`tril`) matrix, (3) softmax with -inf masking, (4) real query/key/value self-attention. The second half adds multi-head attention, feed-forward, residuals, LayerNorm, dropout and scale-up.

**Most useful ideas for slides:**
1. **"Attention is a weighted average of the past."** The `tril` matrix trick shows causal masking in one picture: a lower-triangular matrix with rows that sum to 1.
2. **Communication vs computation:** attention = tokens talk; the MLP = each token thinks by itself. A block is `x = x + sa(ln1(x)); x = x + ffwd(ln2(x))`.
3. **Training targets are the input shifted by one.** Every position is a "predict the next token" example.
4. **Generation loop:** crop to `block_size`, take the last logits, softmax, `multinomial` sample, append, repeat.
5. **Seeing the loss fall as pieces are added.** Bigram about 2.5, then about 2.08 after attention/blocks, then 1.48 after scaling up.

**Numbers (from `gpt.py`):** batch_size 64; block_size (context) 256; max_iters 5000; eval_interval 500; learning_rate 3e-4; n_embd 384; n_head 6 (head size 64); n_layer 6; dropout 0.2; MLP expansion 4x with ReLU; weights initialised with std 0.02.
**Losses (from the third-party summary):** about 2.5 (bigram), 2.08 and then 2.07 before scale-up, **1.48 validation loss** after scale-up.
**nanoGPT README:** the Shakespeare-char config (6 layers, 6 heads, 384 channels, context 256) reaches a best val loss of 1.4697 in about 3 minutes on one A100. The CPU "baby" config (4 layers, 4 heads, 128 dimensions, block 64) gets 1.88. GPT-2 sizes and OpenWebText losses: gpt2 124M (val 3.12), gpt2-medium 350M (2.84), gpt2-large 774M (2.67), gpt2-xl 1558M (2.54). Reproducing GPT-2 124M takes about 4 days on 8x A100 40GB and reaches about 2.85.
**Not verified:** vocab size 65, initial loss about 4.17, parameter count about 10M, training time in the video. The summary I read did not contain them.

### 3.4 "The Annotated Transformer" (Harvard NLP; 2022 update by Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, Stella Biderman; original by Sasha Rush)

**What it covers:** the 2017 paper "Attention Is All You Need", line by line, as working PyTorch code. It includes training on a toy copy task and on German-English (Multi30k).

**Most useful ideas for slides:**
1. **The formulas, exactly:** `softmax(QK^T/sqrt(d_k))V`, the sinusoidal position formula, and multi-head attention.
2. **`subsequent_mask`:** the causal mask as an upper-triangular matrix that is inverted, with masked scores set to -1e9.
3. **Pre-norm vs post-norm, in one line:** the paper says `LayerNorm(x + Sublayer(x))`, but the code does `x + Dropout(Sublayer(LayerNorm(x)))` (pre-norm) "for code simplicity". This is a nice bridge to "what changed since 2017".
4. **Generator = linear + log_softmax;** `greedy_decode` picks the top token in a loop.
5. **Training tricks:** warmup learning-rate schedule, label smoothing.

**Numbers (base model):** N = 6 layers each in encoder and decoder; d_model 512; d_ff 2048; h = 8 heads; d_k = d_v = 64; dropout 0.1; label smoothing 0.1; warmup 4000 steps; Adam beta1 0.9, beta2 0.98, eps 1e-9; about 25,000 source + 25,000 target tokens per batch. Data: WMT14 EN-DE about 4.5M sentence pairs, about 37,000 BPE tokens; EN-FR 36M sentences, 32,000 word-pieces. Hardware: 8x P100. Base model 100,000 steps (about 12 h); big model 300,000 steps (about 3.5 days). BLEU: EN-DE 28.4, EN-FR 41.0 (big model). Toy copy task: vocabulary 11, N = 2, 20 epochs.

### 3.5 Sebastian Raschka, "The Big LLM Architecture Comparison" (Jul 19, 2025; updated Apr 2, 2026)

**What it covers:** a model-by-model tour of open-weight LLM architectures (DeepSeek V3, OLMo 2, Gemma 3, Mistral Small 3.1, Llama 4, Qwen3, SmolLM3, Kimi K2, gpt-oss, GLM-4.5, Qwen3-Next, MiniMax-M2, and others). His main point: the overall shape is still the GPT-style decoder from 2019. What changed is the details.

**Most useful ideas for slides:**
1. **"Same skeleton, new parts":** absolute positions became RoPE, MHA became GQA, GELU became SwiGLU (his intro). LayerNorm became RMSNorm (his OLMo 2 section).
2. **GQA vs MLA:** two ways to shrink the KV cache. GQA shares K/V heads. MLA compresses K/V into a smaller space before caching.
3. **MoE:** many expert MLPs, with a router picking a few per token. Huge total size, small active size.
4. **Sliding-window attention:** some layers only look at nearby tokens, which saves KV-cache memory.
5. **Norm placement:** Post-Norm (2017), Pre-Norm (GPT-2 onward), OLMo 2's Post-Norm variant + QK-Norm, Gemma 3 with norms both before and after.

**Numbers by model (all from the article):**
- **DeepSeek V3/R1:** 671B total, 37B active; MLA; 256 experts per MoE module, 9 active (1 shared + 8 routed), each 2,048 hidden; MoE in every block except the first 3; 61 blocks.
- **Llama 4 Maverick:** 400B total, 17B active; GQA; 2 active experts of 8,192 hidden each; MoE alternates with dense blocks. DeepSeek V3 is about 68% larger in total parameters.
- **Qwen3 MoE:** 235B-A22B (22B active) and 30B-A3B; 8 active experts, **no shared expert** ("8 instead of 8+1 experts"); GQA. The total expert count of 128 is only implied, in the gpt-oss comparison.
- **Qwen3 dense:** 0.6B, 1.7B, 4B, 8B, 14B, 32B. Compared with Llama 3 1B, Qwen3 0.6B is "deeper" (more blocks), while Llama 3 is "wider" (more heads). Qwen3 uses less memory but generates fewer tokens per second. His Qwen3 GQA code has a `qk_norm` option.
- **Gemma 3 (27B; also 1B/4B/12B):** GQA + sliding window, local:global ratio 5:1, window shrunk from 4096 (Gemma 2) to 1024; RMSNorm both before and after attention. Gemma 2 used 1:1 with a 4k window.
- **Mistral Small 3.1 (24B):** plain GQA, sliding window off by default.
- **OLMo 2:** Post-Norm variant (RMSNorm after attention/MLP, inside the residual) + QK-Norm; MHA.
- **SmolLM3 (3B):** NoPE (no positional encoding) in roughly every 4th layer.
- **Kimi K2:** 1T parameters, DeepSeek-V3-style with MLA, more experts and fewer MLA heads.
- **gpt-oss:** 120b and 20b (about 3.6B active); 32 experts with 4 active; 24 blocks, width 2,880; sliding window in every other layer; attention bias; attention sinks.
- **GLM-4.5:** 355B (Air 106B); 3 dense layers before the MoE blocks; shared expert.
- **Qwen3-Next 80B-A3B:** Gated DeltaNet + Gated Attention at 3:1; shared expert added back; native 262k context.
- **MiniMax-M2:** about 10B active; 4.37% of parameters active vs 9.36% for Qwen3 235B.

---

## 4. Five questions

### Q1. How does text turn into numbers?
In two steps: tokenize, then embed. A tokenizer cuts the text into tokens and replaces each one with an integer ID. Karpathy's toy model uses single characters (Karpathy, `gpt.py`); real models use sub-word pieces, such as the about 37,000 BPE tokens of the 2017 model (Annotated Transformer). Each ID then picks a row from a learned embedding table, a vector of `d_model` numbers: 512 in 2017 (Alammar), 12,288 in GPT-3 (3Blue1Brown), 2,048 in your Qwen3-1.7B. 3Blue1Brown's point is that this first vector means the word *without context*, so "mole" starts out the same in every sentence. The blocks then refine it using the rest of the text. Position also has to be added, because attention alone ignores order. 2017 added sine waves (Alammar, Annotated), Karpathy adds a learned position table, and Qwen3 uses RoPE (Raschka).

### Q2. What does attention do, in one sentence?
Attention lets each token look back at the earlier tokens, decide which ones matter to it (query-key scores, then softmax), and pull in a weighted mix of their information (values), updating its own vector. Karpathy frames it as "a weighted average of the past" that the model learns. 3Blue1Brown shows it as adjectives ("fluffy", "blue") updating the noun ("creature"). Alammar's example is "it" attending to "the animal". In Qwen3-1.7B, 16 heads do this in parallel in each of the 28 layers, each with its own idea of what is "relevant". To save memory, pairs of query heads share one of 8 key/value heads (grouped-query attention, as described by Raschka).

### Q3. What does the MLP (feed-forward) part do?
After attention has mixed information *between* tokens, the MLP processes each token *on its own*: it widens the vector, applies a non-linearity, and narrows it back. In 2017 that is 512 to 2,048 to 512 (Annotated). In Karpathy's model it is 4x wider with ReLU. Karpathy calls attention "communication" and the MLP "computation": the token "thinks about" what it just gathered. It also holds a large share of the parameters. 3Blue1Brown notes that attention is only about a third of GPT-3's parameters; much of the rest is in the MLPs. Modern models, Qwen3 included, use the SwiGLU form of the MLP (Raschka). Mixture-of-Experts models replace this single MLP with many expert MLPs and run only a few per token (Raschka: DeepSeek V3 runs 9 of 256). Your Qwen3-1.7B is a dense model, so it uses one MLP per layer.

### Q4. How does the model pick the next word?
The final vector of the last token goes through a linear layer that gives one score (logit) for every token in the vocabulary. Alammar's example has 10,000 words, so 10,000 scores. Softmax turns the scores into probabilities that sum to 1 (Alammar, Annotated). Then a decoding rule picks one token. Greedy decoding takes the highest probability (Alammar; Annotated's `greedy_decode`). Sampling draws at random by probability (Karpathy's `torch.multinomial`). Beam search keeps a few candidate sequences (Alammar). Temperature divides the logits before softmax: below 1 the output is more focused and repeatable, above 1 it is more varied. That formula is standard but not written out in these sources. When you run Qwen3-1.7B locally, the temperature setting in your runner controls exactly this step. The chosen token is appended and the whole process runs again for the next token.

### Q5. Why does it keep a KV cache?
Text is generated one token at a time, and each new token's attention needs the keys and values of *all* earlier tokens. Those do not change, so recomputing them every step would waste work. The model stores them in a cache instead. Raschka's article is mostly about shrinking this cache: GQA shares K/V heads, MLA compresses K/V, and sliding windows keep only nearby tokens in some layers. 3Blue1Brown notes that attention cost grows with context length, which is the other side of the same problem. For your Qwen3-1.7B: per token the cache holds 2 (K and V) x 28 layers x 8 KV heads x 128 dims x 2 bytes = 114,688 bytes = **112 KiB**. At the default 40,960-token context that is **4,480 MiB**, which is why your first run ran out of GPU memory. Capping the context at 8,192 tokens brings it to **896 MiB**. GQA already halves the cache: with 16 K/V heads instead of 8 it would be twice as large.

---

## 5. What changed since 2017 (mainly from Raschka)

| Change | 2017 Transformer | Today | Who uses it (per Raschka unless noted) |
|---|---|---|---|
| **Decoder-only** | Encoder + decoder (Alammar, Annotated) | One decoder stack with a causal mask | Every model in his comparison; Karpathy covers encoder vs decoder at 01:42:39 |
| **Positions** | Sine/cosine added to embeddings | **RoPE** (rotate Q and K by position) | "Absolute to rotational (RoPE)" in his intro; MiniMax-M2 uses *partial* RoPE; SmolLM3 skips positions (NoPE) in every 4th layer |
| **Normalization** | Post-Norm LayerNorm | **Pre-Norm RMSNorm** (from GPT-2 on) | Most models; OLMo 2 uses a Post-Norm variant; Gemma 3 uses norms before *and* after; **QK-Norm** in OLMo 2 and MiniMax-M2, and an option in his Qwen3 code |
| **MLP activation** | ReLU | **SwiGLU** (replaced GELU) | Standard now; Grok 2.5 has a SwiGLU shared-expert-like block |
| **Attention heads** | Full multi-head (MHA) | **GQA**: query heads share K/V heads | Llama 4, Qwen3, Gemma 3, Mistral Small 3.1, gpt-oss, GLM-4.5, SmolLM3; also your Qwen3-1.7B (16 Q / 8 KV) |
| **MLA** | n/a | Compress K/V before caching | DeepSeek V3/R1, Kimi K2; the DeepSeek-V2 ablation he cites found MLA slightly better than MHA and GQA slightly worse |
| **MoE** | One MLP per layer | Many expert MLPs, router picks a few | DeepSeek V3 (256 experts, 9 active, 37B of 671B active), Llama 4 Maverick (17B of 400B), Qwen3 235B-A22B and 30B-A3B (8 active, no shared expert), gpt-oss (32 experts, 4 active), GLM-4.5, Kimi K2 |
| **Sliding window** | Every token sees all earlier tokens | Some layers see only a local window | Gemma 3 (5:1 local:global, window 1024), Gemma 2 (1:1, 4k), gpt-oss (every other layer); Mistral Small 3.1 has it off |

Scale changed too: 2017 base model 6 layers x 512 wide (Annotated), GPT-2 from 124M to 1558M parameters (nanoGPT README), GPT-3 96 layers x 12,288 wide (3Blue1Brown), DeepSeek V3 671B total (Raschka).

---

## 6. Sources (URLs actually read)

1. Jay Alammar, "The Illustrated Transformer": https://jalammar.github.io/illustrated-transformer/ (opened)
2. 3Blue1Brown, "Attention in transformers, step-by-step", **written lesson**: https://www.3blue1brown.com/lessons/attention (opened; the video itself was not watched)
3. Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out", not watched; these were used instead:
   - YouTube description and chapters: https://www.youtube.com/watch?v=kCc8FmEb1nY
   - Lecture code README: https://github.com/karpathy/ng-video-lecture
   - Lecture code: https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/gpt.py
   - nanoGPT README: https://github.com/karpathy/nanoGPT
   - Third-party AI summary with timestamps (loss numbers): https://videohighlight.com/v/kCc8FmEb1nY
4. "The Annotated Transformer" (Harvard NLP, 2022 version): https://nlp.seas.harvard.edu/annotated-transformer/ (opened)
5. Sebastian Raschka, "The Big LLM Architecture Comparison": https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison (opened; the Gemma 4 section mentioned in the update note did not come through in what I read)
