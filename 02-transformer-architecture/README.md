# Task 2: How a transformer picks the next word

**Finding:** A transformer turns text into token IDs, looks each ID up as a vector, passes the vectors through a stack of identical blocks (attention, then an MLP, joined by residual connections), and ends with one score per vocabulary word. The model has to run that whole loop once per generated token, so it keeps the earlier tokens' keys and values in a cache. For Qwen3-1.7B that cache costs **112 KiB per token**: 4,480 MiB at the model's default 40,960-token context, 896 MiB when capped at 8,192. That is the reason my Task 1 test ran out of GPU memory until I capped the context.

## The question

What happens inside a transformer between typing a prompt and getting the next word, and how do a model's settings decide how much memory it needs?

## What I did

1. **Hands-on (done by me with Claude's help):** [inspect_model.py](inspect_model.py) reads the Qwen3-1.7B config and tokenizer (no weights needed) and prints the architecture settings. It then works out the KV cache size from them: 2 (keys and values) x layers x KV heads x head size x 2 bytes.
2. **Reading (sources summarized by Claude, notes reviewed by me):** Claude read the five sources and wrote [notes/readings.md](notes/readings.md): Alammar's *The Illustrated Transformer*, the Annotated Transformer, Raschka's *The Big LLM Architecture Comparison*, and, for the two videos, substitutes. For 3Blue1Brown it used the written lesson with the same title instead of the video. For Karpathy's *Let's build GPT* it used the video description, the lecture code, the nanoGPT README and a third-party summary, not the full video. I have not watched either video. The notes file says this plainly, and marks every number it could not check.
3. **Five questions (answered by Claude from those sources):** how text turns into numbers, what attention does, what the MLP does, how the next word is picked, and why the model keeps a KV cache. They are in section 4 of the notes.
4. **Slides (drafted by Claude):** 18 slides with speaker notes, about 20 minutes: [How a Transformer Works](https://claude.ai/artifact/PQBcTGMzjp6VE16DbtNRf3). I still need to skim them and present them myself.

## Results

Qwen3-1.7B, from `inspect_model.py`:

| Setting | Value |
|---|---|
| Layers (transformer blocks) | 28 |
| Hidden size (numbers per token) | 2,048 |
| Query heads / key-value heads | 16 / 8 (grouped-query attention) |
| Head size | 128 |
| KV cache per token | 112 KiB |
| KV cache at the default 40,960-token context | 4,480 MiB |
| KV cache at 8,192 tokens | 896 MiB |

## What I learned

- **Attention communicates, the MLP computes.** Attention moves information between tokens, and the MLP then processes each token on its own. Each is added to the token's vector through a residual connection.
- **Generation is a loop.** Logits, softmax, pick one token, append, repeat. This is why the cache exists.
- **Grouped-query attention is already saving memory.** With one key-value head per query head the cache would be twice as big.
- **Most of what changed since 2017 is in the details.** Raschka's point: decoder-only layout, RoPE, RMSNorm before each part, SwiGLU, GQA, mixture of experts and sliding windows, on the same skeleton.

## Caveats

- The reading and the slides are Claude's work from the sources it could open. Treat the slide text as a draft to check, not as my own understanding.
- Two items in the notes are marked "not verified": the temperature formula (standard, but none of the five sources writes it out) and some details of Karpathy's video that the sources I could reach did not contain.
- The slide examples (token IDs, the attention grid, the sample sentence) are illustrations and not output from a real model run.

## Still open

- Watch the 3Blue1Brown video and Karpathy's lecture myself, and fix anything in the notes that they contradict.
- Skim the slides, rewrite the parts that are not in my own words, and present them.
- A written version of the explanation for this README (the notes file is the current draft).
