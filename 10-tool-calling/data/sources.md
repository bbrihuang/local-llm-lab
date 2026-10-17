# Sources used (Task 10, tool-calling and agent benchmarks)

Labels: S = self-reported by the model maker, I = independent, A = aggregator (unconfirmed against primary), P = primary paper or project page.

## BFCL
- https://gorilla.cs.berkeley.edu/leaderboard.html (P; version history, scoring wording, "Last Updated 2026-04-12", commit f7cf735, bfcl-eval 2025.12.17; leaderboard table did not load for my fetch tool)
- https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html (P; multi-turn categories, 200 entries each, state and response checks, tool domains)
- https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html (P; 200 web search, 465 memory, 5,200 format-sensitivity entries, DuckDuckGo tools, failure injection, category weights 40/30/10)
- https://openreview.net/forum?id=2GmDdhBdDk (ICML 2025 paper page; blocked by a verification screen, not read)
- https://benchmarklist.com/benchmarks/bfcl_v4/ (A; Claude Opus 4.5 FC 77.47)
- https://benchlm.ai/benchmarks/bfcl-v4 (A; BTL-3 88.5; 23 of 24 rows self-reported; conflicts with the line above)

## tau-bench / tau2 / tau3
- https://arxiv.org/abs/2406.12045 and https://arxiv.org/pdf/2406.12045 (P; 115 retail + 50 airline tasks, gpt-4-0613 user at temperature 1.0, pass^k, gpt-4o 61.2 retail and 35.2 airline pass^1)
- https://arxiv.org/abs/2506.07982 (P; tau2, telecom, dual control)
- https://github.com/sierra-research/tau2-bench (P; domains, tau3 changes, 75+ task fixes, v1.0.1 banking regrade, example user LLM gpt-4.1)
- https://taubench.com (P; leaderboard, timeline, 50+ task fixes 2026-02, Qwen3.5-397B-A17B 87.9)
- https://arxiv.org/pdf/2605.08545 (I; 25 of 50 airline tasks flawed, pass^5 20.8 to 40.0)
- https://www.alphaxiv.org/abs/2512.07850 (SABER; annotation errors; numbers there come from an AI overview, not used)
- https://benchlm.ai/benchmarks/tau2-bench (A; ~99% telecom rows, saturation flag, not comparable across setups)

## GAIA
- https://arxiv.org/abs/2311.12983 (P; 466 questions, human 92% vs GPT-4 plugins 15%)
- https://huggingface.co/datasets/gaia-benchmark/GAIA (P; gated, three levels, file attachments, 450+)
- https://hal.cs.princeton.edu/gaia (I; validation 165 questions, top 74.55, level scores, cost overstatement note)
- https://huggingface.co/spaces/gaia-benchmark/leaderboard (P; submission page only, no scores shown to my tool)
- https://pricepertoken.com/leaderboards/benchmark/gaia (A; bare-model scores, GPT-5 Mini 44.8)
- https://agentmarketcap.ai/blog/2026/04/10/gaia-benchmark-2026-general-ai-agent-performance-test (A, secondary; H2O.ai above 75% test, Level 1 82%; unconfirmed)

## BrowseComp
- https://arxiv.org/abs/2504.12516 (P; 1,266 questions, authors)
- https://openai.com/index/browsecomp/ (P; construction, human 29.2%, GPT-4o 0.6, o1 9.9, Deep Research 51.5)
- https://benchlm.ai/benchmarks/browsecomp (A; top 92.5, all 57 rows self-reported, saturation flag, comparability notes)
- https://anthropic.com/engineering/eval-awareness-browsecomp (I; 9 leaked problems, 2 decrypted answer keys, 86.57 vs 86.81)

## MCP-Universe and MCPBench
- https://arxiv.org/abs/2508.14704 (P; 11 servers, 6 domains, evaluators, GPT-5 43.72)
- https://github.com/SalesforceAIResearch/MCP-Universe (P; domains, agent types)
- https://mcp-universe.github.io/ (P; 231 tasks, leaderboard top 44.59; page inconsistent with itself)
- https://arxiv.org/abs/2504.11094 (P; "Evaluation Report on MCP Servers", introduces MCPBench, Bing 64%)
- https://github.com/modelscope/MCPBench (P; tasks, metrics, open-sourced 2025-04-14)

## Terminal-Bench 2.0
- Reused from Task 9: ../../09-coding-benchmarks/data/sources.md and ../../09-coding-benchmarks/cards.md
- https://benchlm.ai/benchmarks/terminalBench2 (A; GPT-5.5 82% on 2026-10-16 per my fetch tool; the Task 9 card has 84.7% from the Terminal-Bench leaderboard, which is a different source)

## Model reports (report_scores.csv)
- Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B (BFCL-V4 50.3, TAU2-Bench 79.9; no BrowseComp, Terminal-Bench, MCP or GAIA rows)
- Qwen3.5 family: https://huggingface.co/Qwen/Qwen3.5-397B-A17B (BFCL, TAU2, Tool Decathlon, MCP-Mark, BrowseComp, Terminal Bench 2; footnotes on airline fixes and context folding)
- DeepSeek-V4: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro (Terminal Bench 2.0, BrowseComp, MCPAtlas, Toolathlon; no BFCL, tau, GAIA; no separate technical report found)
- GLM-5: https://arxiv.org/pdf/2602.15763 (Table 7; no BFCL, MCP-Universe, GAIA) and https://huggingface.co/zai-org/GLM-5 (not re-opened this task)
- Nemotron 3 Super: https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning (only PinchBench 85.6%; comparison chart is an image), https://research.nvidia.com/labs/nemotron/Nemotron-3-Super (no numbers; points to the technical report PDF, not opened)
