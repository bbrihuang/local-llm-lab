# Readings: what four sources say about agentic and tool-calling evaluation

Fetched 2026-10-17 with a page-summarizing fetch tool (a small model answers a prompt about each page, so quotes are paraphrase-grade). Figures below appear in the fetched text; anything else says "not found".

## 1. Hugging Face Evaluation Guidebook, 2025 dive (yearly_dives/2025-evaluations-for-useful-models.md)
- Page fetched: the tool-calling and assistant-tasks sections. The README says the guidebook is no longer maintained; a newer version is at huggingface.co/spaces/OpenEvals/evaluation-guidebook (not fetched).
- TauBench: the user is simulated by an LLM, so it is "quite costly to run and prone to errors", yet widely used because it reflects real use.
- BFCL has four subsets (single-turn, real-life user calls, multi-turn, agentic: web search, memory, SQL). Scoring mixes AST matching, execution response and state matching. v3 targets tool calling, v4 targets web and search.
- ToolBench mixes mock and real APIs so failures can be accidental; StableToolBench fixes that with a virtual API server but adds an LLM judge (another bias).
- MCP benchmarks (MCPBench, MCP-Universe, LiveMCPBench) depend on live APIs and LLM judges, which causes network failures and reproducibility problems. MCP-Universe checks dynamic tasks against the latest correct answer from the source, which the author likes. LiveMCPBench: "Best models are already reaching 80%", close to saturation.
- Assistant tasks (GAIA, BrowseComp, GAIA2 and others) are scored on whether the end result is correct, not which tools were used. GAIA level 1 is saturated, level 3 still hard; reported numbers vary by method (public validation vs LLM judge on private test).
- BrowseComp: questions are built backward from a known answer, so uniqueness is not guaranteed; the author suspects it is harder.
- GAIA2: time-sensitive and deliberately noisy subsets (mocked failing API calls) are the hardest.
- No scores for any model were in the fetched summary.

## 2. GLM-5 paper (arXiv 2602.15763)
- The ar5iv page was truncated before the evaluation tables. The PDF fetch returned Table 7 and the settings (Sections 4.2.4, 6.1.2, 6.1.3, Appendix B.3). BFCL: not found in the paper text I got.
- BrowseComp has two rows: 62.0 (discard-all) and 75.9 "w/ Context Manage" (keep-recent-k, k=5, plus discard-all once context exceeds 32k tokens). Section 4.2.4 says keep-recent-k alone moves GLM-5 from 55.3 to 62.0. The text appears inconsistent about which row is "the" headline, so I list both.
- BrowseComp judge: official OpenAI prompt with o3-mini, chosen because it matched human annotators best.
- tau2-Bench: small prompt adjustment for Retail and Telecom to avoid premature user termination; Airline uses fixes from the Claude Opus 4.5 system card. The user simulator is not described in the text I got.
- MCP-Atlas: all models re-evaluated on the 500-task public set, per-task timeout extended from 4 to 10 minutes, Gemini 3 Pro as judge.
- Terminal-Bench 2.0 is run with two harnesses (Terminus-2 and Claude Code); a "verified" version fixes ambiguous instructions (60.7 and 61.1 for GLM-5).
- Vending-Bench 2 (simulated vending business over a year, scored on final balance): GLM-5 $4,432 vs Claude Opus 4.5 $4,967.

## 3. DeepSeek-V4 (arXiv 2606.19348 and the Hugging Face model card)
- Paper: two MoE models, V4-Pro (1.6T total, 49B active) and V4-Flash (284B total, 13B active), 1M-token context. The PDF text I got stopped in pre-training, so the paper's own agentic tables were not seen. Numbers below are from the model card (self-reported).
- The paper says V4-Pro-Max is "on par with" Kimi-K2.6 and GLM-5.1 on public agent benchmarks and slightly behind frontier closed models (paper summary, no figures seen).
- Reasoning mode matters a lot: V4-Flash BrowseComp is 53.5 (High) vs 73.2 (Max); Terminal Bench 2.0 for V4-Pro goes 59.1 (Non-Think), 63.3 (High), 67.9 (Max).
- V4-Pro MCPAtlas is 74.2 at High and 73.6 at Max, so more thinking does not always help.
- The model card does not list the harness or settings for these tables, and has no tau2 or BFCL score. It recommends temperature 1.0, top_p 1.0.
- A separate "Agents-last-exam" export (openclaw harness) shows V4-Pro Overall Pass Rate 12.39, noting unattempted tasks count as zero (146 of 152 attempted).

## 4. NVIDIA, Introducing Nemotron 3 Super (developer blog; plus the NVIDIA news blog)
- Model: 120B total, 12B active hybrid Mamba-Transformer MoE, native 1M context.
- Only agentic score found: PinchBench (agent benchmark for OpenClaw) 85.6% "across the full test suite", called best open model in its class. No settings given.
- BFCL, tau, Terminal-Bench, SWE-Bench, BrowseComp, MCP: not found in either page. A bar chart vs GPT OSS 120B and Qwen3 122B exists but the text has no names or values.
- Training for agents: RL in 21 environment configurations, about 1.2M environment rollouts (NeMo Gym, NeMo RL). Claim, no metrics: it "behaves reliably under multi-step workflows" and "reduces reasoning drift".
- The news blog claims "high-accuracy tool calling" over large function libraries, qualitative only. The model also claims "up to 3x" faster generation for code and tool calls via multi-token prediction.
- Takeaway for me: the blog frames tool calling as reliability across many steps, which my 1-turn test does not measure.

## Agentic scores seen (self-reported by the model makers unless noted)

| Model | Benchmark | Score | Source |
|---|---|---|---|
| GLM-5 | BrowseComp | 62.0 / 75.9 (with context mgmt) | GLM-5 paper Table 7 |
| GLM-5 | tau2-Bench | 89.7 | GLM-5 paper |
| GLM-5 | MCP-Atlas (public) | 67.8 | GLM-5 paper |
| GLM-5 | Tool-Decathlon | 39.2 | GLM-5 paper |
| GLM-5 | Terminal-Bench 2.0 | 56.2 (Terminus-2), 56.2 (Claude Code) | GLM-5 paper |
| Claude Opus 4.5 | tau2 / MCP-Atlas / Terminal-Bench 2.0 (Terminus-2) | 91.6 / 65.2 / 59.3 | GLM-5 paper |
| DeepSeek-V3.2 | tau2 / MCP-Atlas | 85.3 / 62.2 | GLM-5 paper |
| DeepSeek-V4-Flash Max | BrowseComp / Terminal-Bench 2.0 / MCPAtlas | 73.2 / 56.9 / 69.0 | V4 model card |
| DeepSeek-V4-Pro Max | BrowseComp / Terminal-Bench 2.0 / MCPAtlas | 83.4 / 67.9 / 73.6 | V4 model card |
| GLM-5.1 Thinking | BrowseComp / Terminal-Bench 2.0 / MCPAtlas | 79.3 / 63.5 / 71.8 | V4 model card |
| Nemotron 3 Super | PinchBench | 85.6% | NVIDIA blog |
| Any of the four | BFCL, GAIA | not found | all four sources |

BrowseComp and Terminal-Bench are not comparable across rows: context strategy, harness and reasoning mode differ.

## What this means for my results
My numbers: Q4_K_M 71/75, Q2_K-imat 60/75, Q2_K 35/75, imat Q2_K copying the calculator example into weather requests.

- Q4_K_M 71/75 is not comparable to any score above. These are multi-step, simulated-user or live-tool tasks; mine is 15 single-turn requests with fake tools. Source says: harness, judge, timeouts and context strategy change scores (GLM-5 BrowseComp 62.0 vs 75.9). My guess: my own prompt wording is such a "harness" too.
- The 4 misses at Q4_K_M (answering 15% of 2,480 itself). Source says: agentic benchmarks score the end result, not which tool was used (HF guidebook). My guess: my strict tool-must-be-called scoring is the opposite design, so Q4_K_M's true "task success" is probably 75/75-like.
- Q2_K never making two calls in one reply. Source says: nothing on quantization in any of the four pages. My guess: multi-call replies need more careful structured output, which heavy compression damages first.
- imat Q2_K recovering to 60/75. Source says: not found. Nvidia's NVFP4 claim ("no loss in accuracy") is about a different format on a big model, so it says nothing about 2-bit GGUF. My guess only.
- The calculator-example copying. Source says: not found anywhere. The closest idea is NVIDIA's "reduces reasoning drift" claim, but that is about long workflows. My guess: a weakened model leans on the nearest literal text in the prompt.
- Reasoning mode matters. Source says: DeepSeek-V4-Flash BrowseComp 53.5 (High) vs 73.2 (Max), Terminal Bench 49.1 (Non-Think) vs 56.9 (Max). My guess: my thinking-off setting may understate even Q4_K_M on harder tasks; I did not test thinking on.
- Run-to-run noise. Source says: HF guidebook flags reproducibility problems with tau-bench (LLM user simulator) and live-API MCP benchmarks. My guess: my 5 tries at temperature 0.7 are noisy in the same way, only milder.

## To verify
- GLM-5 paper: BFCL anywhere? Which BrowseComp row is the headline? What user simulator did tau2 use? (Fetch truncated or text unclear.)
- DeepSeek-V4 paper Section 5.3 agentic tables, and whether they match the model card.
- Nemotron 3 Super technical report: tool-calling/BFCL numbers and the evaluation recipe on GitHub.
- The newer Evaluation Guidebook Space: does it add scores or newer tool-calling benchmarks?
- Whether any source reports tool-calling accuracy by quantization level (none seen); my own Q8_0 and Q4_K_M-imat runs would help.
- Rerun my test with a fairer scorer (accept correct answers without the tool) and thinking on.
