# Readings: notes for Task 7 (inference engines)

How I got these: I fetched each source with a web tool that summarizes pages, so quotes and figures come from those fetched summaries, not from reading every page myself. The PagedAttention figures come from the ar5iv HTML copy of the paper. Anything I did not see is marked "not found".

## 1. Inside vLLM: Anatomy of a High-Throughput LLM Inference System (Aleksa Gordic, 5 Sep 2025)

- It walks through vLLM's V1 engine, from one offline engine up to a multi-node server. It analyzes one specific code commit (42172ad), so details may drift in newer versions.
- The scheduler keeps a `waiting` queue and a `running` queue (first come first served, or by priority) and asks a KV-cache manager for memory. The manager keeps a queue of free blocks. The default block size is 16 tokens.
- Continuous batching: after every single step, the scheduler can add new requests and drop finished ones. The batch is flattened into one long sequence, so requests at different stages share a step.
- At startup, a profiling forward pass decides how big the KV cache can be (it respects `gpu_memory_utilization`; the post's example value is 0.8). Then the attention backend is picked and CUDA graphs are captured, unless `--enforce-eager` is set. This is startup work that llama.cpp mostly skips.
- Chunked prefill splits long prompts so one request cannot hog a step. Prefix caching hashes 16-token blocks and reuses matching ones, and it is on by default.
- Benchmark section: it explains a "roofline" idea. Below some batch size, a step takes about the same time (limited by memory bandwidth). Above it, step time grows roughly with batch size.
- Figures I did not find: a default for `max_num_batched_tokens` (the fetch said none given), and any T4 or llama.cpp comparison.

## 2. Introduction to ggml (Hugging Face blog, 13 Aug 2024)

- ggml is a C/C++ machine learning library focused on running Transformers. llama.cpp and whisper.cpp are built on it, and the article says tools like ollama, jan, LM Studio and GPT4All use it.
- Selling points listed: the core is "less than 5 files", the compiled binary is "less than 1MB" (versus hundreds of MB for PyTorch), it compiles with just GCC or Clang when no GPU is used, and it runs on x86_64, ARM, Apple Silicon, CUDA and others.
- Quantized tensors are supported, to save memory and sometimes to speed things up. The article only name-checks quantization and the GGUF file format as future topics, so it does not explain them.
- You describe the math as a computation graph (`ggml_cgraph`), then hand it to a backend (CPU, CUDA, Metal, Vulkan, RPC) to run. A graph allocator (`ggml_gallocr`) reuses memory for intermediate results based on how long each one lives.
- A scheduler (`ggml_backend_sched`) can use several backends at once, for example sending operations the GPU lacks to the CPU.
- Stated downsides: not every operation exists on every backend, it can need low-level programming knowledge, and breaking changes are expected.
- The article does not cover llama.cpp itself (no server, no slots, no batching), so it does not directly explain my benchmark numbers.

## 3. PagedAttention paper (Kwon et al., SOSP 2023, arXiv 2309.06180)

- Problem: each request's KV cache is large and grows token by token. Older systems reserved one contiguous chunk per request, which wasted memory and limited batch size.
- Measured waste: "20.4% - 38.2%" of KV cache memory held real token data in existing systems.
- Fix: split the KV cache into fixed-size blocks that do not need to sit next to each other, like virtual-memory pages in an operating system. vLLM is the serving system built on this. The abstract says it cuts KV waste to near zero.
- Size example (OPT-13B): about 800 KB of KV cache per token, up to 1.6 GB per request at 2048 tokens. On an A100 40GB, the paper lists 26 GB weights and 12 GB KV cache (about 15.7K token slots).
- Sharing: copy-on-write at block level lets parallel samples and beam search share the prompt's blocks. The paper says beam search can share up to 55% of memory. If memory runs out, a request can be swapped to CPU RAM or recomputed.
- Results: 2 to 4 times the throughput of FasterTransformer and Orca at similar latency. Gains are bigger with longer sequences and bigger models, and smaller when the workload is compute-bound (Alpaca with OPT-175B). Hardware was A100s; models were OPT-13B/66B/175B and LLaMA-13B.
- Not found: block size in the evaluation, and PagedAttention's kernel speed overhead (Sections 7.2 and 7.3 were not in the text I got). The paper compares against older systems, not llama.cpp, and never tests a T4.

## 4. nano-vllm (GeeeekExplorer/nano-vllm on GitHub)

- A from-scratch, small Python version of vLLM: about 1,200 lines per the README, MIT license. It supports prefix caching, tensor parallelism, Torch compile and CUDA graphs. Its API copies vLLM's, with minor differences in `LLM.generate`.
- README benchmark (RTX 4070 Laptop 8GB, Qwen3-0.6B, 256 sequences, random lengths 100 to 1024): vLLM 133,966 output tokens in 98.37 s = 1361.84 tok/s; nano-vllm 93.41 s = 1434.13 tok/s. That is the author's own test on one small model.
- Layout I saw: `nanovllm/` (package), `bench.py`, `example.py`, `pyproject.toml`. I could only read three files inside the package: `engine/block_manager.py`, `engine/scheduler.py`, `config.py`. The GitHub tree page and API were not fetchable, so the full package layout is not confirmed.
- Block manager: blocks are chained by hash (xxh64, each hash includes the previous block's hash), with reference counts so blocks can be shared. Config default `kvcache_block_size` is 256 tokens (vLLM's default is 16 per source 1).
- Scheduler: two queues (`waiting`, `running`). It does prefill first; decode steps only run when no prefill was scheduled. If KV blocks run out it preempts a sequence, frees its blocks and puts it back at the front of `waiting`, so it must redo its prefill. Chunked prefill is limited to the first sequence.
- Config defaults seen: `max_num_batched_tokens` 16384, `max_num_seqs` 512, `max_model_len` 4096, `gpu_memory_utilization` 0.9.
## 5. Defeating Nondeterminism in LLM Inference (Thinking Machines Lab, Sep 2025)

- Main claim: the same prompt at temperature 0 can give different answers mostly because the batch size changes with server load, and common GPU kernels are not "batch invariant" (their math order changes with batch size). The popular "GPU concurrency plus floating point" explanation is called incomplete.
- Background: floating-point addition is not associative, so adding the same numbers in a different order can change the result slightly. Atomic adds are rarely used in an LLM forward pass, so the same input at the same batch size is repeatable.
- Experiment: Qwen3-235B-A22B-Instruct-2507, temperature 0, 1000 completions of "Tell me about Richard Feynman" gave 80 unique outputs. The first difference appeared at token 103. With batch-invariant kernels, all 1000 were identical.
- Three operations needed fixing: RMSNorm, matmul (avoid split-K, one configuration for all shapes, about 20% slower than cuBLAS) and attention (fix the split size, update the KV cache before attention).
- Speed cost (Qwen-3-8B, 1000 sequences, 90 to 110 output tokens): vLLM default 26 s, deterministic 55 s, deterministic with an improved attention kernel 42 s.
- Caveats from the post: results are repeatable on one hardware and software setup, not across different GPUs or library versions. The kernels are not heavily optimized.
## What this explains in my results

Each item is labelled. "Source says" means I saw it in the source. "My guess" means it is my reasoning and untested.

**A. llama.cpp beating vLLM at 32 users on the T4 (281 vs 236 tok/s)**
- Source says (PagedAttention): the big wins come from KV memory waste and bigger batches, with the best gains on longer sequences and bigger models, and smaller when compute-bound. Source says (Gordic): below a saturation batch size, step time is bandwidth-bound and flat.
- My guess: my test is short (about 400 in, 256 out) with a 4B model and only 32 users, so memory waste is not what limits either engine. That removes most of what PagedAttention offers. The README's explanation (the T4 forced vLLM onto the slower `TRITON_ATTN` backend) fits with Gordic's note that the attention backend is chosen at startup, but none of the sources test a T4. Neither source tests llama.cpp at all.

**B. The 1-user limit (about 40 tok/s on the T4, 83 on the 5070; 70 measured on the 5070)**
- Source says (Gordic): small batches are bandwidth-bound, the roofline idea. Source says (PagedAttention): weights are 26 GB in its 13B example and each step must touch them (my reading; the paper stresses memory more than bandwidth).
- My guess: this matches the README's own math (8.05 GB weights divided into memory bandwidth). The remaining gap to the ceiling (70 of 83) is probably launch overhead and KV cache reads, untested.

**C. Round 2 at half speed in llama.cpp**
- Source says: nothing directly. None of the five sources describes llama-server slots or this symptom. Gordic and nano-vllm describe vLLM style scheduling, which has no fixed "slots", and vLLM never showed the problem.
- My guess: the README's idea (which slot new requests land in, perhaps prefix or cache reuse across rounds) is plausible but untested. Another guess: the half-speed is a different batching or scheduling state after round 1. Nano-vllm's scheduler shows how different the two designs are: it runs prefill and decode in separate steps and preempts by recompute, so a round's behavior depends on queue state.

**D. Time to first token arriving in clumps about 1 s apart**
- Source says (Gordic): chunked prefill splits long prompts so no request hogs a step, and prefill and decode compete for the same step budget. Source says (nano-vllm): `max_num_batched_tokens` defaults to 16384 there; I did not find vLLM's default.
- My guess: this fits the README's chunks of about 2,048 tokens. It is consistent, not proven.

**E. 16 users on the 5070 and the 12 GB memory limit**
- Source says (PagedAttention): KV cache grows with every token and a 13B model needs about 800 KB per token. Source says (nano-vllm): when blocks run out, it preempts and recomputes.
- My guess: Qwen3-4B is smaller (I did not compute its per-token KV size), but with 8.05 GB of weights on a 12 GB card, the spare room for KV cache is small, which is why 16 users was the cap. I did not verify this against logs.

**F. Greedy decoding and "same answer" checks**
- Source says (Thinking Machines): temperature 0 does not guarantee the same text when the batch size differs. If I ever compare outputs between 1 user and 32 users, or between engines, small differences are expected. This is relevant to my test because the same prompt text sits at different batch sizes.
- My guess: my speed numbers are not affected, because every request generates exactly 256 tokens regardless of content.

## Open questions / to verify

1. Does llama-server's slot or cache setup explain the slow round 2? Try a short warm-up round, vary `--parallel`, or turn off cache reuse, and compare. Check the llama-server docs; none of my sources cover it.
2. Is the T4 gap really the attention backend? Re-run vLLM with a different backend if one is available, or run on an A100/H100 and see if vLLM pulls ahead at 32 users.
3. Check vLLM's real defaults for block size and `max_num_batched_tokens` on this version (0.31.0). Gordic's post is from Sep 2025 and was written against an older commit.
4. Compute Qwen3-4B's KV cache bytes per token from its config, then see how many users fit on the 5070 beside the 8.05 GB weights.
5. Test longer prompts and outputs, where PagedAttention is supposed to help most.
6. Does batch size change llama.cpp's text at temperature 0? Compare outputs for the same prompt at 1 vs 16 users.
7. Read the unfetched parts: PagedAttention Sections 7.2 and 7.3 (block size, overhead), the rest of the nano-vllm package (model runner, attention code), and the ggml follow-up posts on quantization and GGUF.
