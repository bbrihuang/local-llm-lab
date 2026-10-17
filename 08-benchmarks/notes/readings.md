# Readings: what the benchmark literature says about my Qwen3.5-4B results

Notes written in plain language. Figures appear only if they were in the fetched text. Fetched with WebFetch on 2026-10-17. Some pages were read through a summarizing tool, so exact numbers should be rechecked against the originals (see "To verify").

## 1. Hugging Face Evaluation Guidebook

Pages: "2025 evaluations for useful models" (yearly_dives/2025-evaluations-for-useful-models.md) and "some evaluation datasets" (contents/automated-benchmarks/some-evaluation-datasets.md) in github.com/huggingface/evaluation-guidebook. The repo README says the guidebook is no longer maintained and the latest version (Dec 2025) is on the OpenEvals Space.

- Older reasoning and commonsense sets (ARC 2018, WinoGrande 2019, HellaSwag 2019, CommonsenseQA, PIQA, OpenBookQA) are described as once hard but now too easy, contaminated or saturated. The guide says they are mainly useful for ablations and pretraining checks. It also says big crowdsourced datasets contain errors and low-quality questions.
- MMLU (2020) is called saturated and contaminated, with incomplete questions, wrong answers, ambiguous items and a US-centric topic mix. MMLU-Redux (2024) is a cleaned version.
- MMLU-Pro (2024) is the main community replacement: harder questions and more answer options. Used mostly for pretraining and ablations.
- GPQA (2023) is PhD-level biology, chemistry and physics. The Diamond subset is the most used, but the guide says it has started to show contamination.
- HLE has 2.5K expert-written questions, mostly private, "not broken" so far. No fast automatic scoring exists, so many people use an LLM judge, and results from different sources are often not comparable.
- The guide expects knowledge benchmarks to fade: questions get too complex for non-experts to check, and models with web access turn knowledge tests into search tests.
- The datasets page itself does not cover MMLU-Pro, GPQA or HLE. It lists MMLU as 15K multiple-choice questions ("a strong/high-quality baseline") and ARC as 8K grade-school science questions, and warns that many public datasets are probably contaminated but can still carry signal.
- The guide prefers functional, checkable tests over model judges. It says perplexity analysis is "not very good" at detecting contamination.

## 2. LiveBench (arXiv 2406.19314)

Introduction read from the arXiv PDF (ar5iv failed to convert).

- Problem: test-set contamination. Static benchmark questions leak into newer models' training data, so scores stop measuring real ability and benchmarks go obsolete quickly.
- Evidence cited in the intro: models do worse on Codeforces problems released after their training cutoff, and pre-cutoff performance tracks how often a problem appears on GitHub. A hand-written GSM8K-style variant suggests several models overfit to GSM8K.
- Why common fixes fall short: LLM judges make mistakes and favor their own answers and longer answers; human judges bring formatting and tone bias; human-written questions can be narrow or badly built.
- LiveBench's answer has three parts: frequently updated questions from recent sources, automatic scoring against objective ground truth (no LLM judge), and a diverse set of hard tasks.
- Figures seen: 18 tasks in 6 categories (math, coding, reasoning, language, instruction following, data analysis); questions added and updated monthly; no model above 70% accuracy at the time of writing. The abstract adds that open models tested range from 0.5B to 405B.
- Note: the Qwen3 report uses a "LiveBench 2024-11-25" release as one of its benchmarks, so it is a benchmark Qwen itself reports.

## 3. The Leaderboard Illusion (arXiv 2504.20879)

Introduction read from ar5iv. It is about Chatbot Arena (human-preference voting), not about knowledge quizzes like MMLU-Pro.

- The authors audited about 2M battles covering 243 models from 42 providers (Jan 2024 to Apr 2025). Several authors are at Cohere, which they disclose.
- Undisclosed private testing: some providers test many model variants privately and publish only the best, and scores can be retracted. Meta tested 27 private models before Llama 4. The authors say best-of-N selection inflates scores and their simulations support this.
- Data access gap: Google and OpenAI received an estimated 19.2% and 20.4% of test prompts. 41 fully open-source models together got an estimated 8.9%.
- Using Arena data in fine-tuning raised ArenaHard win-rate from 23.5% to 49.9% when the Arena share went from 0% to 70%, meaning gains can come from fitting the test distribution.
- Silent deprecation: 205 of 243 public models were silently deprecated, only 47 officially listed; 64% of the silently deprecated ones are open-weight or open-source.
- Five recommendations: ban score retraction, limit private variants, apply removals equally across licenses, fair sampling, publish the list of removed models.

## 4. Adding Error Bars to Evals (Miller, arXiv 2411.00640)

Introduction read from ar5iv.

- Core message: report uncertainty, not just a single average score.
- Five practices: (1) standard error of the mean via the Central Limit Theorem; (2) clustered standard errors when questions come in related groups; (3) variance reduction by resampling answers and using next-token probabilities; (4) paired differences when comparing two models (test per-question differences, not two separate averages); (5) power analysis to check whether an eval can detect the gap you care about.
- The intro's example table compares two made-up models, "Galleon" and "Dreadnought", on MATH, HumanEval and MGSM, with differences of +2.5, -3.1 and -2.7 points. Those numbers are fictional and only illustrate that small gaps need error bars.
- The fetch tool noted that HumanEval for Dreadnought is 87.7% in the intro table but 86.7% in Section 2. Not resolved.
- Exact formula for the standard error was not in what I read; my README uses sqrt(p(1-p)/(n-1)), which is the usual form (not confirmed from this paper in this session).

## 5. Report tables

### Which reports I found

| Report | Found? | Source used |
|---|---|---|
| Qwen3 technical report, arXiv 2505.09388 | Yes | ar5iv full text |
| DeepSeek-V4, "Towards Highly Efficient Million-Token Context Intelligence", arXiv 2606.19348 | Abstract found on arXiv. Benchmark tables were read from the Hugging Face model card for DeepSeek-V4-Pro, which links to that report. I did not see the report's own PDF tables. | huggingface.co/deepseek-ai/DeepSeek-V4-Pro |
| GLM-5, "From Vibe Coding to Agentic Engineering", arXiv 2602.15763 | Yes | arxiv.org/pdf/2602.15763 (Table 7) |

Qwen3.5-4B (my model) has no row in any of these three reports. The Qwen3.5 model card's 79.1% for MMLU-Pro is quoted from my README, not re-fetched.

### Knowledge benchmarks seen (only values I saw)

Qwen3 report, post-trained models, thinking / non-thinking:

| Model | MMLU-Redux (think / non-think) | GPQA-Diamond (think / non-think) |
|---|---|---|
| Qwen3-4B | 83.7 / 77.3 | 55.9 / 41.7 |
| Qwen3-8B | 87.5 / 79.5 | 62.0 / 39.3 |
| Qwen3-14B | 88.6 / 82.0 | 64.0 / 54.8 |
| Qwen3-32B | 90.9 / 85.7 | 68.4 / 54.6 |
| Qwen3-235B-A22B | 92.7 / 89.2 | 71.1 / 62.9 |

Qwen3 report, base (pre-trained) models, 5-shot: Qwen3-4B-Base MMLU 72.99, MMLU-Redux 72.79, MMLU-Pro 50.58, SuperGPQA 28.43, GPQA 36.87. Qwen3-8B-Base MMLU-Pro 56.73, Qwen3-235B-A22B-Base MMLU-Pro 68.18. MMLU-Pro is not in the Qwen3 post-trained tables I saw. HLE: not found in Qwen3.

DeepSeek-V4 (HF model card table, attributed to the report), reasoning modes:

| Benchmark | V4-Flash Non-Think | V4-Flash Max | V4-Pro Non-Think | V4-Pro Max |
|---|---|---|---|---|
| MMLU-Pro | 83.0 | 86.2 | 82.9 | 87.5 |
| GPQA Diamond | 71.2 | 88.1 | 72.9 | 90.1 |
| HLE | 8.1 | 34.8 | 7.7 | 37.7 |
| SimpleQA-Verified | 23.1 | 34.1 | 45.0 | 57.9 |

Same table, other models vs V4-Pro Max (87.5 MMLU-Pro, 90.1 GPQA-D, 37.7 HLE): Opus-4.6 Max 89.1 / 91.3 / 40.0; GPT-5.4 xHigh 87.5 / 93.0 / 39.8; Gemini-3.1-Pro High 91.0 / 94.3 / 44.4; GLM-5.1 Thinking 86.0 / 86.2 / 34.7. Base models, 5-shot MMLU-Pro: V3.2-Base 65.5, V4-Flash-Base 68.3, V4-Pro-Base 73.5.

GLM-5 report (Table 7, GLM-5 vs others): GPQA-Diamond GLM-5 86.0, GLM-4.7 85.7, DeepSeek-V3.2 82.4, Claude Opus 4.5 87.0, Gemini 3 Pro 91.9, GPT-5.2 (xhigh) 92.4. HLE (no tools) GLM-5 30.5, Opus 4.5 28.4, Gemini 3 Pro 37.2, GPT-5.2 35.4. HLE with tools GLM-5 50.4. MMLU-Pro: not found in the GLM-5 report.

What stands out: the same benchmark is reported with very different settings (thinking vs not, tools vs not, shots, base vs post-trained). Qwen3-4B GPQA-Diamond is 55.9 thinking vs 41.7 non-thinking in the same table, a 14-point swing from one switch. HLE for DeepSeek-V4-Pro is 7.7 non-think vs 37.7 Max.

## What this means for my results

Each link is labelled. "Source says" means a reading above supports it; "my guess" means I am reasoning beyond the sources.

1. ARC 95 / 96 is very high for a 4B model.
   - Source says: the Evaluation Guidebook lists ARC (2018) among the older sets that are now too easy, contaminated or saturated, mostly used for ablations. (Guidebook)
   - My guess: my 95 to 96% is consistent with that, and part of it may be memorized training data rather than reasoning. I did not test for contamination.

2. MMLU-Pro 70 to 74 vs the official 79.1.
   - Source says: results depend heavily on setup. The Qwen3 and DeepSeek tables show big swings between thinking and non-thinking modes and between shot counts (DeepSeek-V4-Pro MMLU-Pro 82.9 non-think vs 87.5 Max; Qwen3-4B GPQA-D 41.7 vs 55.9). (Qwen3, DeepSeek-V4)
   - Source says: the Evaluation Guidebook notes that comparing numbers from different sources is often not apples to apples (said there about HLE and LLM judges).
   - My guess: my prompt, 4,096 and 14,000 token caps, 210-question subset and Q4_K_M quantization each explain some of the 5 to 9 point gap. I cannot split it between them.

3. Plus or minus 3 points of noise.
   - Source says: Miller recommends reporting a standard error and, when comparing two models or settings, testing paired per-question differences, not two averages. A power analysis tells you whether an eval can detect a given gap. (Error Bars)
   - Source says (illustrative only): the paper's fictional example shows gaps of 2.5 to 3.1 points between models, the size of my thinking-vs-not differences.
   - My guess: my thinking-on vs off gaps (2.4 and 3.8 points) are within about one standard error each (3.0 to 3.2), so I should not claim thinking helps on MMLU-Pro. Because both settings use the same 210 questions, a paired comparison would be more sensitive than comparing the two averages. I did not do it.

4. 14 to 18 of 210 replies with no final answer.
   - Source says: nothing I read covers truncated replies directly. DeepSeek and Qwen tables show thinking modes matter a lot, which is related. (Not found in the sources: any statement about token caps or unanswered replies.)
   - My guess: unanswered replies count as wrong in my scoring, so this is a scoring choice, and it likely accounts for a large part of the gap to 79.1. Official numbers may use larger token budgets or different handling. I do not know which.

5. Why not trust a single leaderboard number.
   - Source says: LiveBench argues static test sets become contaminated, and Leaderboard Illusion argues that private testing and best-of-N selection can inflate public scores on Chatbot Arena. (LiveBench, Leaderboard Illusion)
   - My guess: the Leaderboard Illusion is about Arena votes, not MMLU-Pro, so it only applies loosely to the model card number. Still, a company-reported score is a best-case, self-run number, and my README already marks some top scores as self-reported.

6. MMLU-Pro and GPQA are still the standard picks, with caveats.
   - Source says: the Guidebook calls MMLU-Pro the main replacement for MMLU and says GPQA Diamond has started to show contamination. DeepSeek-V4 and GLM-5 both still report GPQA-Diamond and HLE. (Guidebook, DeepSeek-V4, GLM-5)
   - My guess: for a 4B model, GPQA-Diamond and HLE are too hard to give a useful signal at 3 questions each, so my 30-question table says little.

## To verify

- LiveBench intro: the 18 tasks, 6 categories and "70%" figures came from a summarizing tool reading the PDF. Check the PDF text.
- Leaderboard Illusion: the final 742 characters of the ar5iv page were not read. Re-check the 19.2%, 20.4%, 8.9%, 23.5% to 49.9%, 205/243 and 47 figures.
- Error Bars: confirm the standard-error formula and whether the paper uses n or n-1, and resolve the 87.7 vs 86.7 mismatch.
- DeepSeek-V4: confirm Tables 1 to 3 in the report PDF itself. I only saw the model-card copy and the arXiv abstract. The arXiv ID 2606.19348 came from the model card page.
- Qwen3.5 model card: confirm 79.1% MMLU-Pro, which setting (thinking or not) and which prompt it used. The Qwen3.5-4B model is not in any of the three reports I read.
- Qwen3 report: confirm Table values against the PDF (ar5iv text was summarized by a tool, one table was truncated after MATH-500).
- GLM-5: check whether any MMLU-Pro or other knowledge table exists beyond Table 7.
- Evaluation Guidebook: the web Space version (OpenEvals) may differ from the GitHub markdown I read.
- Try a paired test on my two 210-question MMLU-Pro runs (off vs on) to see if thinking really differs.
- Rerun a small MMLU-Pro sample with a larger token cap to see how many of the 14 to 18 unanswered replies become right.
