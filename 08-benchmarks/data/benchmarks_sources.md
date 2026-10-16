# Benchmark sources and notes

Researched 2026-10-15. "Self-reported" means the number comes from the model maker, not an independent run. Anything marked **unconfirmed** could not be traced to a primary source.

## MMLU
- Paper: Hendrycks et al., "Measuring Massive Multitask Language Understanding", arXiv v1 2020-09-07. Authors are from UC Berkeley, Columbia, UChicago and UIUC. https://arxiv.org/abs/2009.03300
- Size: 57 subjects and 15,908 questions. The paper gives the test set as 14,079; the common HF test split has 14,042. Each question has 4 options. Unspecialized MTurk workers scored 34.5%; the paper estimates expert-level accuracy at about 89.8%. https://arxiv.org/pdf/2009.03300
- Saturation path: GPT-4 scored 86.4% (5-shot) in 2023-03 (https://arxiv.org/pdf/2303.08774). Gemini Ultra scored 90.0% using CoT@32 in 2023-12, and Google called it "first to outperform human experts" (https://blog.google/technology/ai/google-gemini-ai).
- Best confirmed score: o3-high at 93.3% in OpenAI simple-evals (https://github.com/openai/simple-evals). llm-stats lists GPT-5 at 92.5%, self-reported, but the primary source is **unconfirmed** (https://llm-stats.com/benchmarks/mmlu).
- Label errors: MMLU-Redux ("Are We Done with MMLU?", 2024-06) estimates 6.49% of questions have errors, including 57% of the analysed Virology questions. https://arxiv.org/abs/2406.04127

## MMLU-Pro
- Paper: Wang et al., arXiv v1 2024-06-03. Authors are from the University of Waterloo, University of Toronto and CMU. It has 12,032 questions across 14 disciplines, each with 10 options. At launch the best model was GPT-4o at 72.6% with CoT. https://arxiv.org/abs/2406.01574 and https://arxiv.org/pdf/2406.01574
- Best independent score, from Artificial Analysis: Gemini 3 Pro Preview (High) at 89.8%, followed by Claude Opus 4.5 (Reasoning) at 89.5%. https://artificialanalysis.ai/evaluations/mmlu-pro
- Self-reported scores: Qwen3.7 Max at 89.6% (https://benchlm.ai/benchmarks/mmlu-pro) and Sakana Namazu at 90.3%, which is **unconfirmed** (https://llm-stats.com/benchmarks/mmlu-pro).
- There is no human baseline. The label-error rate is **unconfirmed**.

## GPQA-Diamond
- Paper: Rein et al., arXiv v1 2023-11-20. Authors are from NYU, Cohere and Anthropic. The full set has 448 questions; Diamond is a 198-question subset. Each question has 4 options. https://arxiv.org/abs/2311.12022 and https://arxiv.org/pdf/2311.12022
- Human baselines on the full set: experts 65% (74% after discounting clear mistakes) and skilled non-experts with web access 34%. On Diamond, experts scored about 81% and non-experts about 22%, but the paper notes these are skewed by how Diamond was selected.
- In 2024-09, o1 scored 77.3% pass@1, and OpenAI called it the first model to surpass PhD experts. https://openai.com/index/learning-to-reason-with-llms/
- In 2025-11, Gemini 3 Pro scored 91.9%. https://china.googleblog.com/2025/11/gemini-3.html
- Best score: GPT-6 Astra at 96.0%, self-reported, 2026-09. https://llm-stats.com/benchmarks/gpqa and https://benchlm.ai/benchmarks/gpqa-diamond
- Best independent score: Gemini 3.1 Pro Preview at 95.45% (Vals AI, page updated 2026-09-01). https://www.vals.ai/benchmarks/gpqa
- Ceiling: Epoch AI (2025-05) estimates about 8% of Diamond questions may be invalid, which puts the ceiling around 90-92%. https://epoch.ai/gradient-updates/gpqa-diamond-whats-left

## Humanity's Last Exam (HLE)
- Paper: Phan et al., arXiv v1 2025-01-24, from the Center for AI Safety and Scale AI. It has 2,500 questions: about 24% multiple choice, the rest exact-match, and about 14% multimodal. o3-mini judges the answers. At launch the best score was o3-mini (high) at 13.4%. There is no human baseline. https://arxiv.org/abs/2501.14249 and https://arxiv.org/pdf/2501.14249
- Best score without tools (confirmed on Anthropic's page): Claude Fable 5.1 at 60.9%, 2026-09. Fable 5 scored 57.8% and Opus 5 scored 56.6%. https://www.anthropic.com/claude-fable-and-mythos-5-1
- Best score with tools: Claude Opus 5.5 at 67.7%, 2026-09-22. https://www.anthropic.com/claude-opus-5-5
- **Unconfirmed:** llm-stats and BenchLM list Claude Opus 5.5 at 64.4% without tools. Anthropic's Opus 5.5 page reports only the with-tools score, and a tracker lists the model's "Chartography (no tools)" score as 64.4%, so this may be a mix-up. https://llm-stats.com/benchmarks/humanity's-last-exam-(no-tools%2C-text-only) and https://benchlm.ai/benchmarks/hlenotools
- Label issues: FutureHouse (2025-07) estimated that about 29% of the text-only chem/bio answers conflict with the literature. The HLE team's own review found about 18% of a bio/chem subset problematic. https://futurehouse.org/research/hle-exam

## ARC (AI2 Reasoning Challenge, Challenge set)
- Paper: Clark et al., arXiv v1 2018-03-14, from AI2. The Challenge set has 2,590 questions (1,119 train, 299 dev, 1,172 test). Questions are mostly 4-option multiple choice. Scoring gives 1/k credit for ties. At launch the best baseline was 27.11%. https://arxiv.org/abs/1803.05457 and https://arxiv.org/pdf/1803.05457
- GPT-4 scored 96.3% (25-shot) in 2023-03. https://arxiv.org/pdf/2303.08774
- Claude 3 Opus scored 96.4% (25-shot) in 2024-03. https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf
- **Unconfirmed:** codesota lists o3 at 98.1% and Gemini 2.5 Pro at 97.8%, but the primary sources it cites do not obviously report ARC. https://www.codesota.com/benchmark/arc-challenge
- The paper gives no human baseline.

## HellaSwag
- Paper: Zellers et al., arXiv v1 2019-05-19, from UW and AI2. It has about 70k problems, with 10K validation and 10K test. Each has 4 endings. Humans scored 95.6% on test; BERT-Large scored 47.3%. https://arxiv.org/abs/1905.07830 and https://arxiv.org/pdf/1905.07830
- GPT-4 scored 95.3% (10-shot) in 2023-03, matching humans (https://arxiv.org/pdf/2303.08774). Claude 3 Opus scored 95.4% in 2024-03 (Claude 3 model card above). https://llm-stats.com/benchmarks/hellaswag
- Errors: Surge AI found problems in 36% of 300 sampled validation rows. https://www.surgehq.ai/blog/hellaswag-or-hellabad-36-of-this-popular-llm-benchmark-contains-errors
- Validity audit: arXiv 2504.07825 found ungrammatical text, typos and options that are equally correct. https://arxiv.org/abs/2504.07825

## WinoGrande
- Paper: Sakaguchi et al., arXiv v1 2019-07-24, from AI2 and UW. It has 43,972 problems (dev 1,267, test 1,767), each with 2 options. Humans scored 94.0%; the best model at launch scored 79.1%. https://arxiv.org/abs/1907.10641 and https://arxiv.org/pdf/1907.10641
- Best score: ST-MoE-32B (Google), fine-tuned, 96.1% on the dev set, 2022-02. https://arxiv.org/pdf/2202.08906 and https://vercel.hyper.ai/en/sota/tasks/common-sense-reasoning/benchmark/common-sense-reasoning-on-winogrande
- Best few-shot LLM score: GPT-4 at 87.5% (5-shot), 2023-03. https://arxiv.org/pdf/2303.08774 and https://llm-stats.com/benchmarks/winogrande

## SimpleQA
- Paper: Wei et al., arXiv v1 2024-11-07, from OpenAI. It has 4,326 short-answer questions graded correct, incorrect or not attempted, plus an F-score. The estimated dataset error rate is about 3%. A third trainer scored 94.4% on 1,000 questions. At launch, o1-preview scored 42.7% correct. https://arxiv.org/abs/2411.04368 and https://arxiv.org/pdf/2411.04368
- Best score without search: ERNIE 5.0 at 74.01% (Table 2, 2026-02, self-reported; the paper does not state tool use). Gemini 3 Pro scored 69.33% in the same table. https://arxiv.org/pdf/2602.04705 (llm-stats lists ERNIE 5.0 at 75.0%.)
- From OpenAI: GPT-4.5 scored 62.5% (2025-02). https://openai.com/index/introducing-gpt-4-5/ and https://github.com/openai/simple-evals
- With search: DeepSeek V3.1's search-agent score is 93.4% (https://huggingface.co/deepseek-ai/DeepSeek-V3.1). DeepSeek-V3.2-Exp with tools scores 97.1% (https://llm-stats.com/benchmarks/simpleqa).
- Labs increasingly report SimpleQA Verified instead. Gemini 3 Pro scored 72.1% on it. https://china.googleblog.com/2025/11/gemini-3.html

## IFEval
- Paper: Zhou et al., arXiv v1 2023-11-14, from Google and Yale. It has 541 prompts covering 25 verifiable instruction types. There are 4 metrics: prompt-level and instruction-level, each strict and loose. At launch, GPT-4 scored 76.89% prompt-level strict. https://arxiv.org/abs/2311.07911 and https://arxiv.org/pdf/2311.07911
- Llama 3.3 70B scored 92.1% in 2024-12. https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- Best score: Qwen3.5-27B at 95.0%, 2026-02, self-reported. https://huggingface.co/Qwen/Qwen3.5-27B and https://llm-stats.com/benchmarks/ifeval
- There is no human baseline.

## RULER
- Paper: Hsieh et al., arXiv v1 2024-04-09, from NVIDIA. It has 13 synthetic tasks in 4 categories, run at 4K-128K with 500 examples per task per length. Scoring is recall-based accuracy. A model's "effective length" is the longest length where it scores above 85.6%. At launch, Gemini 1.5 Pro scored a 95.8 average (94.4 at 128K). https://arxiv.org/abs/2404.06654 and https://arxiv.org/pdf/2404.06654
- Leaderboard top: Jamba-1.5-large averages 96.0 (author-reported), Gemini 1.5 Pro 95.8 and Qwen2.5-14B-1M 95.7. https://github.com/NVIDIA/RULER
- **Unconfirmed:** the 2024-08 date for Jamba-1.5-large, which is its release month and is not stated on that page.
