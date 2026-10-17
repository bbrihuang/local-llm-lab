# Sources used (Task 9, coding benchmarks)

Labels: S = self-reported by the model maker, I = independent, A = aggregator (unconfirmed against primary).

## HumanEval
- https://arxiv.org/abs/2107.03374 (original paper; Codex 28.8%, GPT-3 0%)
- https://arxiv.org/pdf/2303.08774 (GPT-4 technical report, 67.0% 0-shot; S)
- https://arxiv.org/abs/2305.01210 and https://github.com/evalplus/evalplus (HumanEval+/MBPP+ test weakness; I)
- https://codesota.com/benchmark/humaneval , https://codesota.com/browse/computer-code/code-generation (top scores; A, vendor-reported)

## MBPP
- https://arxiv.org/abs/2108.07732 (paper; 974 tasks, 59.6% at launch)
- https://github.com/evalplus/evalplus (MBPP+ 399 -> 378 tasks, ground-truth fixes)
- https://codesota.com/browse/computer-code/code-generation (o4-mini 94.9; A)

## LiveCodeBench
- https://arxiv.org/abs/2403.07974 (paper)
- https://github.com/LiveCodeBench/LiveCodeBench (release versions and problem counts)
- https://livecodebench.github.io/leaderboard.html (leaderboard did not load for my fetch tool)
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro (93.5; S)
- https://letsdatascience.com/news/deepseek-v4-preview-adds-open-weight-1m-context-models-4a8f21c6 and https://framia.converge.ai/page/en-US/news/deepseek-v4-announcement-april-2026 (search results suggesting the 2026-04-24 DeepSeek V4 date; I did not open them; unconfirmed)

## SciCode
- https://arxiv.org/abs/2407.13168 , https://scicode-bench.github.io/
- https://artificialanalysis.ai/evaluations/scicode (I)
- https://www.alphaxiv.org/abs/2608.04975 (SciCode-Verified audit)

## SWE-bench Verified
- https://arxiv.org/abs/2310.06770 (SWE-bench paper)
- https://openai.com/index/introducing-swe-bench-verified/ (creation, 2024-08)
- https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/ (2026-02-23 audit)
- https://www.vals.ai/benchmarks/swebench (I; mini-SWE-agent; archived)
- https://benchlm.ai/benchmarks/swe-bench-verified (A; 105 of 111 rows provider self-reports)
- https://www.anthropic.com/news/claude-3-7-sonnet , https://www.anthropic.com/news/claude-opus-4-5 , https://www.anthropic.com/news/claude-opus-5 (S; Opus 5 page lists no SWE-bench Verified)
- https://www.codesota.com/benchmark/swe-bench (A; its timeline dates looked inconsistent so I did not use them)
- https://www.morphllm.com/terminal-bench-2 (secondary report on mini-swe-agent git-history leak and Datacurve audit; unconfirmed)
- https://datanorth.ai/news/claude-opus-5-by-anthropic (secondary; Opus 5 released 2026-07-24, 96%)

## Terminal-Bench 2.0
- https://arxiv.org/abs/2601.11868 (paper, 89 tasks)
- https://www.snorkel.ai/blog/terminal-bench-2-0/ and https://snorkel.ai/leaderboard/terminal-bench-2-0/ (release date, leaderboard)
- https://www.tbench.ai/news/announcement-2-0 (release post; no date shown to my fetch tool, leaderboard page empty)
- https://www.vals.ai/benchmarks/terminal-bench-2 (I)
- https://benchlm.ai/benchmarks/terminalBench2 (A)

## Model reports
- Qwen3: https://arxiv.org/pdf/2505.09388
- Qwen3.5: https://huggingface.co/Qwen/Qwen3.5-397B-A17B , https://huggingface.co/Qwen/Qwen3.5-4B (reachable)
- DeepSeek-V4: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro (I did not find a separate technical report PDF; the model card table was used)
- GLM-5: https://arxiv.org/pdf/2602.15763 , https://huggingface.co/zai-org/GLM-5

## SWE pick (psf__requests-1142)
- https://datasets-server.huggingface.co/search?dataset=princeton-nlp/SWE-bench_Verified&config=default&split=test&query=psf__requests-1142 (membership check)
- https://github.com/psf/requests/issues/1051 (issue), https://github.com/psf/requests/pull/1142 , https://patch-diff.githubusercontent.com/raw/psf/requests/pull/1142.diff (gold patch)
- https://raw.githubusercontent.com/psf/requests/22623bd8c265b78b161542663ee980738441c307/requests/models.py (base code)
