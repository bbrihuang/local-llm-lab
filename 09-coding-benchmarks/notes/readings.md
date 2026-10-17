# Readings: coding benchmarks

Notes from five sources, fetched 2026-10-17. Exact figures appear only where they were in the fetched text. Fetching used a summarizing tool, so every figure is "as returned by the fetch tool", not checked against the PDF myself (see To verify).

## 1. Hugging Face Evaluation Guidebook, "2025 evaluations for useful models" (code section)

Source: github.com/huggingface/evaluation-guidebook, yearly_dives/2025-evaluations-for-useful-models.md. The repo README says the guidebook is no longer maintained; the newer copy is on the HF Space OpenEvals/evaluation-guidebook.

- It lists code benchmarks in rough age order: MBPP (2021, 1K crowdsourced entry-level Python problems), APPS (2021, 10K interview-style problems), HumanEval (2021), HumanEval+ and MBPP+ (EvalPlus, 2023), EvoEval (2024), LiveCodeBench (2024), AiderBench, RepoBench (2023), SWE-bench (2024).
- On HumanEval it says the problems were "specifically made for the release" and that it came with a sandbox for running untrusted code and introduced the pass@k estimator. It gives no contamination or saturation discussion for HumanEval.
- EvalPlus is described as adding test cases, fixing bugs in the originals and adding more inputs. No score numbers are given.
- LiveCodeBench is called an "excellent contamination free benchmark" because it stores problem dates, so you can compare performance on problems from before and after a model's training cutoff.
- SWE-bench: models must solve existing GitHub issues, testing "logic understanding, cross file editing and execution, long context reasoning". The guide recommends the higher-quality Verified subset for post-training.
- Practical advice: coding sets are "good proxies for reasoning"; for post-training use SWE-bench Verified, for comparing models at inference use LiveCodeBench and AiderBench.
- SciCode is listed under science assistants, not code. The guide says models were "quite bad at it at publication (less than 5% scores)" and it does not know current results.
- Terminal-Bench is not mentioned. The guide does not say any code benchmark is saturated.

## 2. LiveBench, arXiv 2406.19314 (abstract and introduction)

The ar5iv page failed to convert, so this comes from the arXiv PDF and abstract page.

- Motivation: benchmarks published online leak into training data ("test set contamination"), and the usual alternatives (LLM judges, human crowdsourcing) bring their own biases.
- Contamination evidence cited: performance on Codeforces problems drops after a model's training cutoff, and pre-cutoff performance tracks how often a problem appears on GitHub.
- Three design properties: frequently updated questions from recent sources, automatic scoring against objective ground truth (no LLM judge), and a diverse set of hard tasks.
- 18 tasks in six categories (math, coding, reasoning, language, instruction following, data analysis). Coding has a code generation task based on LiveCodeBench and a code completion task.
- Questions are refreshed monthly; the paper says about 1/6 are replaced per update, so a full refresh takes about six months.
- Headline: top models score below 70%. Table 1 puts o1-preview-2024-09-12 first at 64.7 overall. Correlation with ChatBot Arena is 0.91 and with Arena-Hard 0.88 (from section 3.3, not the introduction).
- No HumanEval, SWE-bench or Terminal-Bench numbers; this is a freshness argument, not a coding score source.

## 3. Qwen3 technical report, arXiv 2505.09388

- Post-trained models are reported in two modes, thinking and non-thinking, in separate tables.
- Qwen3-4B thinking (Table 17): LiveCodeBench v5 54.2; CodeForces rating 1671 (92.8th percentile).
- Qwen3-4B non-thinking (Table 18): LiveCodeBench v5 21.3; CodeForces 842 (33.7%).
- The thinking jump on the same small model is large: 54.2 vs 21.3 on LiveCodeBench v5.
- HumanEval and MBPP appear only for base models. Qwen3-4B-Base: EvalPlus 63.53 (an average of HumanEval, MBPP, HumanEval+ and MBPP+), MBPP 67.00, MultiPL-E 53.13, CRUX-O 55.00. Separate HumanEval figures: not found.
- SWE-bench, Terminal-Bench, SciCode: not found in the fetched text.
- The report also varies the "thinking budget" on four benchmarks across math, coding and STEM, using Qwen3-235B-A22B (Figure 2). No exact coding numbers or 4B figures were returned.

## 4. DeepSeek-V4 model card (Hugging Face, DeepSeek-V4-Pro page)

I found no separate technical report; this is the model card only. The "Comparison across Modes" table gives three reasoning modes.

- LiveCodeBench pass@1: Flash 55.2 / 88.4 / 91.6 and Pro 56.8 / 89.8 / 93.5 (Non-Think / High / Max).
- SWE-bench Verified resolved: Flash 73.7 / 78.6 / 79.0 and Pro 73.6 / 79.4 / 80.6.
- Terminal-Bench 2.0: Flash 49.1 / 56.6 / 56.9 and Pro 59.1 / 63.3 / 67.9.
- Reasoning mode matters most on LiveCodeBench (Pro: 56.8 to 93.5) and least on SWE-bench Verified (73.6 to 80.6).
- HumanEval pass@1 (0-shot) appears only for base models: DeepSeek-V3.2-Base 62.8, V4-Flash-Base 69.5, V4-Pro-Base 76.8.
- MBPP and SciCode: not found. The card does not state the scaffold or LiveCodeBench version in the fetched text.

## 5. GLM-5 report, arXiv 2602.15763

- Table 7 (agentic coding): SWE-bench Verified 77.8, SWE-bench Multilingual 73.3, Terminal-Bench 2.0 56.2 with the Terminus-2 scaffold and 56.2 with Claude Code (the table also shows 60.7 and 61.1 with a dagger).
- The dagger means a "verified version of TerminalBench 2.0, fixing some ambiguous instructions", published on Hugging Face as zai-org/terminal-bench-2-verified. So the benchmark itself had unclear tasks.
- SWE-bench numbers were run with the OpenHands framework and a prompt tailored for GLM-5. Scaffold is stated in the paper, and it differs per benchmark.
- Same table, other models: Claude Opus 4.5 80.9 on SWE-bench Verified and 59.3 on Terminal-Bench (Terminus-2); GPT-5.2 (xhigh) 80.0 and 54.0; GLM-4.7 73.8 and 41.0.
- LiveCodeBench: not mentioned. SciCode is named only as one of ten Artificial Analysis Index evaluations, with no score. HumanEval appears only in a base-model architecture ablation (values 33.5 to 38.5), not as a GLM-5 result.
- The paper describes thinking modes (interleaved, preserved, turn-level) but gives no per-mode coding scores.

## Coding scores seen in fetched text

| Model | Benchmark | Score | Mode | Source |
|---|---|---|---|---|
| Qwen3-4B | LiveCodeBench v5 | 54.2 | thinking | Qwen3 report, Table 17 |
| Qwen3-4B | LiveCodeBench v5 | 21.3 | non-thinking | Qwen3 report, Table 18 |
| Qwen3-4B | CodeForces rating | 1671 / 842 | thinking / non-thinking | Qwen3 report |
| Qwen3-4B-Base | MBPP, EvalPlus | 67.00, 63.53 | base (not post-trained) | Qwen3 report, Table 7 |
| DeepSeek-V4-Flash | LiveCodeBench | 55.2 / 88.4 / 91.6 | Non-Think / High / Max | V4-Pro model card |
| DeepSeek-V4-Flash | SWE-bench Verified | 73.7 / 78.6 / 79.0 | Non-Think / High / Max | V4-Pro model card |
| DeepSeek-V4-Flash | Terminal-Bench 2.0 | 49.1 / 56.6 / 56.9 | Non-Think / High / Max | V4-Pro model card |
| DeepSeek-V4-Pro | LiveCodeBench | 56.8 / 89.8 / 93.5 | Non-Think / High / Max | V4-Pro model card |
| DeepSeek-V4-Pro | SWE-bench Verified | 73.6 / 79.4 / 80.6 | Non-Think / High / Max | V4-Pro model card |
| DeepSeek-V4-Pro | Terminal-Bench 2.0 | 59.1 / 63.3 / 67.9 | Non-Think / High / Max | V4-Pro model card |
| DeepSeek-V4-Pro-Base | HumanEval | 76.8 | base | V4-Pro model card |
| GLM-5 | SWE-bench Verified | 77.8 | not stated (OpenHands) | GLM-5 report, Table 7 |
| GLM-5 | Terminal-Bench 2.0 | 56.2 | not stated (Terminus-2 and Claude Code) | GLM-5 report, Table 7 |
| o1-preview | LiveBench overall | 64.7 | not stated | LiveBench paper, Table 1 |

Not found in the fetched text: any post-trained model's HumanEval score, any MBPP or SciCode score for these models, GLM-5 LiveCodeBench, Qwen3 SWE-bench or Terminal-Bench.

## What this means for my results

- **HumanEval 83.5% (thinking off) and 80.5% (on) are not directly comparable with these tables.** Source says: none of the three reports lists a post-trained HumanEval score; they use LiveCodeBench, SWE-bench and Terminal-Bench instead. My guess: that is a sign HumanEval no longer separates modern models, which fits the README's claim that it is near its ceiling.
- **Thinking helped a lot in the reports but not in my run.** Source says: Qwen3-4B goes from 21.3 to 54.2 on LiveCodeBench v5 with thinking, and DeepSeek-V4-Pro from 56.8 to 93.5 (Non-Think to Max). My guess: HumanEval is easy enough that the non-thinking model already gets most problems, so thinking has little to add, and my gap (83.5 vs 80.5) is inside noise anyway.
- **The 25 empty replies from the 14,000-token limit.** Source says: the Qwen3 report treats the thinking budget as a setting that changes coding results (Figure 2, for the 235B model; no numbers seen). My guess: my token cap acted as a small thinking budget, and the 95.0% pass rate on the 139 finished answers shows what a larger budget might give. The reports do not test this on HumanEval or on a 4B model, so I cannot confirm it.
- **Same score, different setup.** Source says: DeepSeek reports three modes (Non-Think, High, Max) and GLM-5 reports two scaffolds, so one model has several valid scores. My guess: my two runs are two points on the same kind of curve, and I should always write the mode and token limit next to a score.
- **SWE-bench task psf__requests-1142.** Source says: SWE-bench Verified numbers (77.8 GLM-5, 80.6 V4-Pro Max) were produced by an agent scaffold (OpenHands for GLM-5), and the guide describes SWE-bench as testing cross-file editing and long context. My guess: my one-task experience (the fix is small, finding it is the hard part) matches that description, but I did not run an agent, so I cannot compare my result to any number here.
- **Benchmarks have flaws too.** Source says: GLM-5 had to publish a "verified" Terminal-Bench 2.0 to fix ambiguous instructions (scores 56.2 vs 60.7 on the Terminus-2 scaffold). My guess: this fits the README's point that scores depend on the exact benchmark version, so my own prompt-and-checker HumanEval score should not be compared with published ones.
- **LiveBench and LiveCodeBench.** Source says: both refresh questions to limit contamination, and LiveBench replaces about 1/6 of its questions per update. My guess: HumanEval, public since 2021, is the opposite case, so contamination is a plausible reason for high scores. Neither source here tests this for HumanEval.
- **Where would Qwen3.5-4B land?** The README's 55.8 on LiveCodeBench v6 comes from the model card, which I did not fetch here. My guess: it is close to the Qwen3-4B thinking number (54.2 on v5), but versions differ, so I would not compare them.

## To verify

- Read the PDF tables myself for Qwen3 (Tables 17 and 18), GLM-5 (Table 7) and the V4 card, since the figures here came through a summarizing fetch tool.
- Find out which LiveCodeBench version and which scaffold DeepSeek used for V4, and whether a separate V4 technical report exists.
- Fetch the Qwen3.5-4B model card (the 55.8 figure is only in my README) and check its mode and version.
- Look at the Qwen3 Figure 2 numbers on thinking budget, and test my own 14,000-token cap against a larger one on the 25 empty problems.
- Check GLM-5 for a HumanEval, MBPP or LiveCodeBench result in the full text, not just the abstract page (the summaries said not found, which may reflect the fetch).
- Check whether the HF Space version of the guidebook differs from the GitHub copy (the GitHub copy is stated as unmaintained).
- Look up LiveBench's current coding scores for small models; the 2024 paper has none I could use.
