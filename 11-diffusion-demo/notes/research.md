# DiffusionGemma demo rebuild: research notes (2026-10-17)

Legend: [V] = read on the cited page. [UNVERIFIED] = not confirmed. Several pages were read through a summarizing fetch tool, so exact wording may differ slightly; quotes are short.

## 1. DiffusionGemma model card
Source: https://huggingface.co/google/diffusiongemma-26B-A4B-it (and https://huggingface.co/docs/transformers/main/en/model_doc/diffusion_gemma)

- Model id: `google/diffusiongemma-26B-A4B-it` (instruction-tuned only; no base variant found). [V]
- Architecture: based on Gemma 4 26B-A4B MoE. 25.2B total, 3.8B active, 30 layers, 128 experts (8 active + 1 shared), 256K context, ~550M vision encoder, text+image in, text out. Apache 2.0. [V]
- Design (docs page): encoder-decoder. The autoregressive encoder prefills the prompt and builds the KV cache. The decoder uses bidirectional attention over a 256-token "canvas" and reads the cache via cross-attention. When a canvas is done, the encoder processes it, appends it to the KV cache, and the next canvas starts (multi-canvas, block-autoregressive). [V]
- Canvas length: 256 (`DiffusionGemmaConfig.canvas_length`, default 256). [V]
- Recommended sampling (model card; also https://unsloth.ai/docs/models/diffusiongemma.md): Entropy-Bounded denoising with adaptive stopping; max 48 denoising steps (per canvas); temperature linear decay 0.8 -> 0.4; entropy bound 0.1; non-selected tokens fully renoised; early stop when mean canvas entropy < 0.005 and top predictions match across two consecutive steps. [V]
- Names in transformers `generate` kwargs (docs page): `max_denoising_steps`, `sampler_config=EntropyBoundSamplerConfig(entropy_bound=...)`, `t_min`, `t_max`, `stability_threshold`, `confidence_threshold`, `cache_implementation`, `streamer`, `decoder_input_ids` (starting canvas, shape (bsz, 256)). `top_k` not supported. `use_cache` not accepted. [V] Library defaults are not stated on the docs page [UNVERIFIED]; the model card values above are the recommended ones.
- Speed claim: ~15-20 tokens per forward pass; >1100 tokens/s per user at low batch on H100 with FP8. [V] (card; blog https://huggingface.co/blog/gemma4)
- Thinking: put `<|think|>` at the start of the system prompt (card) / `enable_thinking=True` in apply_chat_template (docs). [V]
- Transformers version: card says only `pip install -U transformers torch accelerate` and "latest version". The docs page is for `main` and says it "requires installation from source", with stable v5.17.0 linked as the pip alternative; model contributed 2026-06-10. [V] So a stable release v5.17.0 may already include it, but I could not confirm which release first shipped DiffusionGemma [UNVERIFIED]. Safe plan: try `pip install -U transformers`, and if `DiffusionGemmaForBlockDiffusion` import fails, `pip install git+https://github.com/huggingface/transformers`. The Space bundles a dev wheel `transformers-5.8.0.dev0+2db78a1296` (https://huggingface.co/spaces/huggingface-projects/diffusiongemma-codegen/tree/main), meaning early on a dev build was needed.
- Note: the docs say Diffusers is the primary home for DiffusionGemma; transformers only gets bug fixes. [V] (Diffusers usage not researched.)
- Card snippet (verbatim, from https://huggingface.co/google/diffusiongemma-26B-A4B-it):
```python
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor
MODEL_ID = "google/diffusiongemma-26B-A4B-it"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
message = [{"role": "user", "content": "Why is the sky blue?"}]
input_ids = processor.apply_chat_template(message, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt").to(model.device)
output = model.generate(**input_ids, max_new_tokens=512)
text = processor.decode(output[0], skip_special_tokens=False)
```
  (Layout condensed; arguments are as on the card. The docs page decodes with `output.sequences[0][input_len:]`, so `output[0]` on the card may differ by version [UNVERIFIED which is right].)
- Streaming drafts (docs page): `from transformers import TextDiffusionStreamer; streamer = TextDiffusionStreamer(tokenizer=processor.tokenizer); model.generate(**inputs, max_new_tokens=256, streamer=streamer)`. If the streamer has `put_draft`, denoising-step tokens go there; final tokens go to `streamer.put`. Docs also suggest `cache_implementation="static"` to trigger torch.compile. [V]
- Memory: the card gives no number. Unsloth's table (https://unsloth.ai/docs/models/diffusiongemma.md): BF16 52 GB, 8-bit 28 GB, 4-bit 18 GB (total RAM+VRAM, GGUF context). bf16 repo is 48.1 GiB per https://huggingface.co/GoedelMachines/diffusiongemma-26B-A4B-w4a16. Weights alone ~50-52 GB in bf16 (25.2B x 2 bytes), so an 80 GB GPU is needed for bf16 with room for KV cache. [V/derived]
- Quantized variants:
  - GGUF: `unsloth/diffusiongemma-26B-A4B-it-GGUF` (Q4_K_M ~16 GB download, Q8_0), needs llama.cpp unmerged PR 24423 and `llama-diffusion-cli`. Not usable from transformers. [V] https://unsloth.ai/docs/models/diffusiongemma.md
  - Community W4A16: `GoedelMachines/diffusiongemma-26B-A4B-w4a16`, ~17 GB, ~18-20 GB VRAM, load with `AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True)`, needs `transformers>=5.12.1`, `torch>=2.11`, triton; tuned for GB10/RTX 5090; first forward ~45 s JIT. Third-party, unvetted code (trust_remote_code). [V] https://huggingface.co/GoedelMachines/diffusiongemma-26B-A4B-w4a16
  - Other community: mmastrac `.dgq` formats (custom runtime), https://huggingface.co/mmastrac/diffusiongemma-q4emb [only seen in search results; UNVERIFIED].
  - Plain bitsandbytes 4/8-bit via transformers: no official guidance found [UNVERIFIED]; MoE expert layers often are not quantized by bnb, so don't assume it works.
  - MLX: not found for DiffusionGemma (the Gemma 4 regular model has `mlx-community/gemma-4-26b-a4b-it-4bit`, blog above).
- Gating: the model card/blog show no access-request step; license Apache 2.0. [V as far as the page text showed] However the Space README says an `HF_TOKEN` secret is needed "for accessing the private model" (https://huggingface.co/spaces/huggingface-projects/diffusiongemma-codegen/raw/main/README.md), likely a pre-release leftover [UNVERIFIED]. Safe: log in to HF anyway (free token) in Colab.

## 2. Regular Gemma 4 26B-A4B
- Ids: `google/gemma-4-26B-A4B-it` (instruction-tuned), `google/gemma-4-26B-A4B` (base). 25.2B total / 3.8B active, 30 layers, 128 experts, BF16, Apache 2.0, no gating shown. https://huggingface.co/google/gemma-4-26B-A4B-it , https://huggingface.co/blog/gemma4 [V]
- Load (card): 
```python
processor = AutoProcessor.from_pretrained("google/gemma-4-26B-A4B-it")
model = AutoModelForMultimodalLM.from_pretrained("google/gemma-4-26B-A4B-it", dtype="auto", device_map="auto")
```
  `pip install -U transformers torch accelerate`. [V]
- Memory: same shape as DiffusionGemma, ~52 GB bf16 (derived; no table on the page). GGUF Q4_K_M: `ggml-org/gemma-4-26b-a4b-it-GGUF`. [V] blog.
- Fair-comparison note: both models have the same weights size; the regular one is an autoregressive decoder, so tokens/s is a fair comparison on the same GPU/precision.

## 3. The Space (huggingface-projects/diffusiongemma-codegen)
Files: README.md, app.py (13.6 kB), index.html (16.9 kB), requirements.txt (`accelerate`, `sentencepiece`, `hf_xet`), and the bundled transformers dev wheel. https://huggingface.co/spaces/huggingface-projects/diffusiongemma-codegen/tree/main
(app.py was read through a summarizing tool, not a raw download; the proxy blocked curl.)
- README front matter: `sdk: gradio`, `sdk_version: 6.17.3`, `python_version: "3.12"`, title "Gemma Diffusion Website Builder". Runs on ZeroGPU ("Running on Zero"). https://huggingface.co/spaces/huggingface-projects/diffusiongemma-codegen/raw/main/README.md
- Mechanism: a custom `QueueDiffusionStreamer` (a `BaseStreamer` subclass). `put_draft(...)` is called each denoising step and queues a `"draft"` frame (decodes `confirmed_ids + ids`); `put(...)` is called when a canvas finishes and queues a `"commit"` frame (first `put` call is the prompt and is skipped); `end()` queues `"end"`. `model.generate` runs in a worker thread under `torch.inference_mode()`; the GPU function drains a `queue.Queue` and yields frames. Decoding: `tok.decode(ids, skip_special_tokens=True)`. https://huggingface.co/spaces/huggingface-projects/diffusiongemma-codegen/raw/main/app.py
- generate kwargs in the Space: `streamer=streamer`, `max_new_tokens` (default 2048), `max_denoising_steps` (default 64, cap 120), optional `confidence_threshold=1e-9` and `stability_threshold=max_iters` ("full denoise"), and `canvas_ids=` for warm-start from the previous page. No temperature/entropy args are set (library defaults). Canvas size comes from `model.config.canvas_length`. Note: the Space's `canvas_ids` differs from the public docs' `decoder_input_ids`; the Space uses a custom wheel, so the name may be old [UNVERIFIED].
- ZeroGPU: `import spaces` before torch; `@spaces.GPU(duration=_estimate_duration, size=GPU_SIZE)`, `GPU_SIZE` env `GDIFF_GPU_SIZE` default `"xlarge"`; duration capped at 120 s. Model loaded at module scope with `dtype=torch.bfloat16, low_cpu_mem_usage=True` then `.to(DEVICE)`. (ZeroGPU "xlarge" is, I believe, a full H200-class slice [UNVERIFIED].)
- Server: `gradio.Server` (FastAPI) with endpoint `@app.api(name="generate", concurrency_limit=1, time_limit=600, stream_every=0.05)`; each yield is a JSON frame with `kind` (draft/commit/done/error), `source`, `block`, `step`, `canvas`, `max_iters`. Final `done` frame carries cleaned HTML.
- UI (index.html): uses the Gradio JS client (`Client.connect(...)`, `c.submit("/generate", payload)`, `for await (const ev of sub)`). Every frame carries the full source snapshot; `renderCode` diffs lines (LCS) against the previous frame and flashes changed lines. Preview: sandboxed iframe (`sandbox="allow-scripts"`), `preview.srcdoc` reassigned on every frame, with `normalizeHtml` to repair a broken opening and a scroll-position-preserving script. Tweaks: previous page reused as starting canvas. https://huggingface.co/spaces/huggingface-projects/diffusiongemma-codegen/raw/main/index.html
- For a Colab rebuild: the simplest equivalent is `TextDiffusionStreamer` subclass overriding `put_draft`/`put`, plus an iframe `srcdoc` in a notebook widget; a library-provided streamer exists so the bundled wheel is probably not needed with current transformers [UNVERIFIED].

## 4. Blog "Welcome Gemma 4" (https://huggingface.co/blog/gemma4)
- DiffusionGemma: discrete diffusion, 256-token canvas, encoder prefill + KV cache, multi-canvas, ~15-20 tokens per forward pass, >1100 tok/s per user at low batch (H100, FP8), adaptive compute (code/structured tasks need fewer steps). Sampler: EB, up to 48 steps, temp 0.8->0.4, entropy bound 0.1. Quality tradeoff vs 26B-A4B: MMLU Pro 77.6 vs 82.6, AIME 2026 69.1 vs 88.3, GPQA Diamond 73.2 vs 82.3, HLE 11.0 vs 8.7. [V]
- The blog has no memory table. [V]

## 5. Colab with Google AI Pro
- Compute units: Google AI Pro gives 200 Colab compute units/month (article dated 2026-09-27), can reach A100 80GB, plus Colab CLI (released 2026-06-05) supporting T4, A100, H100, TPU v5e/v6e. https://www.inside.com.tw/article/42497-google-ai-pro-colab-200-compute-units-a100-cli (secondary source; official numbers not seen [UNVERIFIED]). Official FAQ confirms eligible paid Google AI plans include monthly compute units and more powerful GPUs, and that the exact amounts/rates are not published and change: https://research.google.com/colaboratory/faq.html . Standalone Colab Pro is 100 units/month, up to 53 GB high-RAM VMs (https://aicoolies.com/pricing/google-colab, third party).
- GPUs and rates (third-party measurement, rates checked 2026-03-10, may fluctuate): T4 15 GB, 1.19 CU/h; L4 22.5 GB, 1.71 CU/h; A100 40 GB 5.40 CU/h, A100 80 GB 7.52 CU/h (select with High-RAM option); H100 80 GB listed but rate unknown. Unit price $0.10. https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/ . H100 availability to AI Pro users is only from the secondary article above [UNVERIFIED]; check the runtime menu. Note G4 / RTX PRO 6000 (96 GB) GPUs exist on Colab per some reports, not verified here.
- Session: free up to 12 h; Pro+ background execution up to 24 h; idle timeouts apply; availability not guaranteed (FAQ above). Idle GPU runtimes keep burning units; disconnect promptly.
- What fits: bf16 (~50 GB weights + KV/activations): A100 80GB or H100 80GB only (A100 40GB, L4, T4 cannot). 8-bit (~28 GB): A100 40GB. 4-bit (~17-18 GB): L4 (22.5 GB) is tight but possible; T4 (15 GB) no. But no official 4/8-bit loading path for DiffusionGemma in transformers was found, so for a faithful speed comparison use bf16 on A100 80GB.
- Budget: 200 CU / 7.52 CU/h = ~26 hours of A100 80GB per month (if the rate is still accurate and 200 CU is right).

## 6. Colab issues, disk, Drive
- No DiffusionGemma-specific Colab issue reports found [UNVERIFIED]. Unsloth links an A100 Sudoku fine-tuning notebook for this model, so A100 works.
- Flash attention / bitsandbytes: not mentioned on any page read. Use default attention; skip bitsandbytes. The w4a16 community build requires torch>=2.11 (may need upgrading Colab's torch).
- Disk: ~50 GB bf16 weights. Colab GPU VMs usually have ~100+ GB local disk, but I did not find an official figure [UNVERIFIED]; check `df -h`. Gemma 4 comparison model adds another ~50 GB, so run one, delete cache (or point cache to different dirs), then the other.
- Download speed: the Space requirements include `hf_xet`, i.e. HF's Xet storage client, which is the default for recent huggingface_hub; `hf_transfer` is the older alternative (`HF_HUB_ENABLE_HF_TRANSFER=1`). Speed-up figures for Colab not verified [UNVERIFIED]. Typical Colab download speeds make 50 GB take roughly 5-20 min [UNVERIFIED estimate].
- Google Drive cache: possible (mount Drive, set `HF_HOME` or `cache_dir` to a Drive folder; 5 TB is plenty). Pitfalls (general knowledge, not from a cited page): Drive-FUSE is slow for reading many GB (load can take longer than re-downloading from HF), and writes may sync lazily so copy completion must be verified; mmap-style safetensors loading from Drive is slow; Drive API quota/"too many files" errors can occur. Better: cache on Drive as a one-time copy (`cp -r` from Drive to local `/content`) at session start, or just re-download with Xet. [UNVERIFIED, untested]

## Summary of decisions
- GPU: A100 80GB (7.52 CU/h, ~26 h/month on 200 CU) for bf16 of both models; H100 if offered. Use High-RAM option.
- Precision: bf16 for both (same footprint, fair tokens/s comparison). Quantized options are third-party/GGUF (llama.cpp PR), not a clean transformers path.
- Download: ~52 GB per model (~104 GB for both, one at a time unless disk allows).
- Install: try latest stable transformers first; fall back to git main.
- Blockers/unknowns: official Colab CU rates and whether AI Pro exposes A100 80GB/H100 today (check the runtime selector); gating status (log in to HF to be safe); the Space's `canvas_ids` vs docs' `decoder_input_ids`; free local PC cannot run bf16 (12 GB VRAM).
