# local-llm-lab

Twelve hands-on experiments with local language models, done as part of BU AI Society's Local LLM Division. They were run on one gaming PC (Windows 11, NVIDIA RTX 5070 with 12 GB of memory) with a few Colab GPU sessions for the jobs that needed more. Each folder asks one question, measures it, and says what the numbers do and do not show.

The main model is **Qwen3.5-4B**, with Qwen3 models of other sizes where a task needed them. Most speed tests use **llama.cpp**; everything is in this repo so it can be rerun.

## What I found

| Task | Question | Result |
|---|---|---|
| [1. First local model](01-first-local-model/) | Is reading a prompt the same work as writing an answer? | No. Prefill ran at about 17,300 tokens/s and decode at about 220 tokens/s on Qwen3-1.7B Q8_0, roughly 80x apart. Decode is limited by memory speed. |
| [2. Transformer architecture](02-transformer-architecture/) | What happens between a prompt and the next word, and what does it cost in memory? | The key-value cache costs 112 KiB per token for Qwen3-1.7B: 4,480 MiB at the default 40,960-token context, which is why my first test ran out of GPU memory until I capped the context. |
| [3. Quantization](03-quantization/) | What does shrinking Qwen3.5-4B from 16 bits to 2 bits cost? | 8 bits is free, 4 bits costs a little, plain 2 bits breaks the model (57% same top token as the full model). An importance matrix cut the 2-bit damage about 4x at the same file size. |
| [4. Quantization methods](04-quantization-methods/) | How do LLM.int8(), SmoothQuant, GPTQ, AWQ, QuaRot, FP4 and BitNet compare? | On Qwen3-0.6B (fake quantization, so quality only, not speed): 8-bit methods cost 1.5% or less; 4-bit weights cost about 12% for AWQ and NVFP4; with 4-bit activations too, only NVFP4 stayed usable (+25%). A Hadamard rotation did not help here. |
| [5. Speculative decoding](05-speculative-decoding/) | Does a small draft model speed up a bigger one? | Not on this GPU. The best case was 0.96x the speed of the big model alone, because the draft model is too expensive per guess on a fast card. |
| [6. Attention types](06-attention-types/) | How much of the context does a head actually use? | A typical head gets 90% of its attention from about 34 of 3,500+ earlier tokens, and from layer 3 on the very first token takes 20% to 75% of attention. |
| [7. Inference engines](07-inference-engines/) | llama.cpp against vLLM on the same GPU | On a Colab T4, llama.cpp matched vLLM for 1 to 4 users and led from 8 users (281 against 236 tokens/s total at 32 users). On the RTX 5070, llama.cpp reached 543 tokens/s total at 16 users. |
| [7.5. Agent frameworks](07.5-agent-frameworks/) | Do agent frameworks change the answers? | Four setups gave 23 to 24 of 30 right. The frameworks added 34% to 52% more prompt tokens; smolagents with its default prompt got 7 of 30 with this model. |
| [8. Knowledge benchmarks](08-benchmarks/) | How does a 4B model score, and how fast do benchmarks fill up? | ARC-Challenge 95% to 96%; MMLU-Pro 70% to 74% against Qwen's 79.1%, partly because 14 to 18 of 210 replies were cut off. I scored 12 of 24 on a small quiz against Qwen's 17. |
| [9. Coding benchmarks](09-coding-benchmarks/) | HumanEval and SWE-bench | HumanEval 83.5% with thinking off, 80.5% on (25 replies ran out of tokens). The SWE-bench fix was applied for me by a script; the README says so. |
| [10. Tool calling](10-tool-calling/) | Which compression levels break tool use? | Q4_K_M got 94.7% of tries right, Q2_K 46.7%, and Q2_K with an importance matrix 80.0%. |
| [12. Diffusion explainer](12-diffusion-explainer/) | How does diffusion work, from images to text? | A 33,794-parameter flow-matching model turned noise into a shape; 1 step gave a blob (distance 0.378), 8 steps 0.036, 100 steps 0.026. |
| [11. Diffusion demo](11-diffusion-demo/) | How fast is DiffusionGemma? | 146 to 340 tokens/s against 39.3 for regular Gemma 4 on the same GPU (3.7x on chat, 6.3x to 8.6x on web pages). My baseline was probably slow, so the ratios are generous. |

I did task 12 before task 11 so the demo would make sense. All timings and scores are medians or single runs on my machine, with the caveats in each README.

## Who did what

I ran every command and measurement myself, on my own PC and on Colab. Claude (Anthropic's assistant) helped by writing and explaining the scripts, drafting the reports and the slide decks, and summarizing reading material. Specifically:

- **Scripts and experiments:** written with Claude, run by me, results pasted back by me.
- **Readings:** for Tasks 2 and 4 to 12, the sources were summarized by Claude and I reviewed the notes (`notes/readings.md` in each folder). Those summaries were fetched through a page-summarizing tool, so exact figures should be rechecked against the originals. I have not yet read the sources myself.
- **Slide decks:** drafted by Claude for Tasks 2, 4, 6 and 12. I have not presented them yet.
- **Task 9:** I did not find the SWE-bench fix myself; Claude gave me a script that applied it, and I checked the four cases.
- **Task 8 quiz:** the answers are mine. Claude merged my three terminal logs into `results/my_answers.csv`.

Where a result is unexplained or unverified, the task README says so. Examples: the half-speed second round in Task 7, and the unconfirmed 672 GB/s memory figure in Task 1.

## Reproducing it

Each task folder has a Commands section. Python scripts run in a virtual environment (`.venv`, with torch, transformers, numpy, matplotlib and huggingface_hub). llama.cpp was build b11630. Model files are not in git: the GGUF files are on Hugging Face ([Qwen3.5-4B-GGUF](https://huggingface.co/bbrihuang/Qwen3.5-4B-GGUF), [Qwen3-4B-GGUF](https://huggingface.co/bbrihuang/Qwen3-4B-GGUF)). The GPQA and Humanity's Last Exam questions are private and are not in this repo.

On a 12 GB card, run llama.cpp with `-c 8192`: the default context's cache alone is 4,480 MiB.
