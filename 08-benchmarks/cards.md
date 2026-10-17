# Benchmark cards

One card per benchmark: what it tests, an example, how it is scored, the top score and whether it is maxed out, and how Qwen3.5-4B (Q4_K_M, my RTX 5070) did on my 3 questions. "Off" and "on" mean thinking off and on. Sources for every number: [data/benchmarks_sources.md](data/benchmarks_sources.md). "Self-reported" means the company that made the model published the score and nobody independent has checked it. GPQA and HLE ask that their questions not be posted online, so those two cards describe the format and show no question.

## MMLU (2020-09, UC Berkeley)
- **Tests:** broad academic and professional knowledge in 57 subjects, from math and history to law and medicine. About 15,900 questions.
- **Example:** "Which of these is the smallest unit that natural selection can change?" (four options)
- **Scoring:** accuracy on four-option multiple choice, usually with 5 examples in the prompt.
- **Human baseline:** non-expert crowd workers 34.5%; estimated expert level about 89.8%.
- **Top score:** 93.3% (OpenAI o3, 2025-04). Maxed out since about 2023-12, when Gemini Ultra claimed to beat the expert estimate. A review found about 6.5% of the questions have errors, so the real ceiling is about 93 to 94%.
- **Qwen3.5-4B on my 3:** 2 of 3 off, 2 of 3 on. It missed a question about 40,000-year-old cave art.

## MMLU-Pro (2024-06, University of Waterloo / TIGER-Lab)
- **Tests:** a harder MMLU: 14 subjects, 10 answer options instead of 4, and the trivial or noisy questions removed. 12,032 questions.
- **Example:** "What are three economic roles of government?" (ten options)
- **Scoring:** accuracy, usually with chain-of-thought reasoning.
- **Human baseline:** none reported.
- **Top score:** 89.8% (Gemini 3 Pro Preview, independent run, 2025-11); self-reported scores reach about 90%. Near its ceiling.
- **Qwen3.5-4B:** 3 of 3 off, 3 of 3 on. On the harness (210 questions): 70.0% off, 72.4% and 73.8% on. Qwen's official figure: 79.1%.

## GPQA-Diamond (2023-11, NYU)
- **Tests:** graduate-level biology, physics and chemistry questions written to be "Google-proof": experts get them right, smart non-experts with web access mostly don't. 198 questions (the Diamond subset).
- **Example format:** a paragraph-long science problem with four answer options. (No question shown.)
- **Scoring:** accuracy on four options; random guessing gets 25%.
- **Human baseline:** domain experts about 81% on Diamond, skilled non-experts with web access about 22%.
- **Top score:** 96.0% (GPT-6 Astra, self-reported, 2026-09); best independent run 95.45% (Gemini 3.1 Pro Preview). Maxed out about 2025-11. Epoch AI estimates about 8% of the questions are invalid, which puts the real ceiling around 90 to 92%, so the top scores are at or above it.
- **Qwen3.5-4B:** 2 of 3 off, 3 of 3 on. Qwen's card lists 76.2% (mode not stated).

## Humanity's Last Exam, HLE (2025-01, Center for AI Safety and Scale AI)
- **Tests:** very hard expert-written questions across 100+ subjects, built to be hard for the best models at launch. 2,500 questions; about 14% need an image.
- **Example format:** mostly short exact-answer questions, about a quarter multiple choice. (No question shown.)
- **Scoring:** accuracy, graded by an AI judge against the reference answer.
- **Human baseline:** none reported.
- **Top score:** 60.9% without tools (Claude Fable 5.1, Anthropic-reported, 2026-09); 67.7% with tools (Claude Opus 5.5). **Not maxed out.** Launch best was 13.4%. Estimates say 18% to 29% of the answers in the chemistry and biology parts may be wrong, so the real ceiling is below 100%.
- **Qwen3.5-4B:** 1 of 3 off. On: 0 of 3 right, one answer wrong and two I haven't judged yet. One off answer ran out of its 4,096 tokens.

## ARC-Challenge (2018-03, Allen Institute for AI)
- **Tests:** grade-school science questions that simple word-matching and search programs get wrong. 2,590 in the Challenge set.
- **Example:** "A red giant star differs from main sequence stars because it ..." (four options)
- **Scoring:** accuracy, usually with 25 examples in the prompt.
- **Human baseline:** none reported in the paper.
- **Top score:** 96.4% (Claude 3 Opus, 2024-03). Maxed out since 2023-03 (GPT-4 96.3%). Websites list 97 to 98%, but I couldn't find a source for them.
- **Qwen3.5-4B:** 3 of 3 both ways. Harness (200 questions): **95.0% off, 96.0% on.**

## HellaSwag (2019-05, University of Washington and AI2)
- **Tests:** commonsense: pick the most plausible next sentence for an everyday scene. The wrong endings were written by a program and filtered to fool earlier models.
- **Example:** "Sharpening knives: A cooker talks in a kitchen and shows the inside of a sharpener knife that has discs inside. then ..." (four endings)
- **Scoring:** accuracy, usually with 10 examples in the prompt.
- **Human baseline:** 95.6%.
- **Top score:** 95.4% (Claude 3 Opus, 2024-03). Maxed out since 2023-03 (GPT-4 95.3%). A company that checked found errors in 36% of 300 sampled rows, so the last few points are mostly mistakes in the test.
- **Qwen3.5-4B:** 2 of 3 off, 3 of 3 on.

## WinoGrande (2019-07, AI2 and University of Washington)
- **Tests:** pronoun and fill-in-the-blank questions that need common sense, built in large numbers and filtered against word-association tricks. About 44,000 questions.
- **Example:** "Dennis made their weight loss goal before Eric, because _ only ate a healthy diet." (Dennis or Eric)
- **Scoring:** accuracy on two options.
- **Human baseline:** 94.0%.
- **Top score:** 96.1% (ST-MoE-32B, fine-tuned on the training set, 2022-02), above the human baseline. The best score from a model with just a few examples in its prompt is GPT-4's 87.5%. Maxed out since 2022-02.
- **Qwen3.5-4B:** 3 of 3 both ways.

## SimpleQA (2024-11, OpenAI)
- **Tests:** short fact questions with a single clear answer, to measure how much a model knows and whether it admits when it doesn't. 4,326 questions.
- **Example:** "What year did Kinoko Teikoku release their first single 'Taikutsu Shinogi'?"
- **Scoring:** an AI grader marks each answer correct, incorrect or not attempted.
- **Human baseline:** none formal; a check of 1,000 questions by a third person agreed about 94% of the time.
- **Top score:** about 74% (ERNIE 5.0, self-reported, 2026-02) without searching the web. **Not maxed out** closed-book; with web search, scores reach 90 to 97%.
- **Qwen3.5-4B:** 1 of 3 off, 0 of 3 on. It named the wrong discoverer of a minor planet, said "approximately 250-260 songs" for a number whose real answer is 137, and got a year wrong (2014 instead of 2012).

## IFEval (2023-11, Google and Yale)
- **Tests:** following instructions that a program can check, like "at least 400 words", "no commas", "exactly 2 paragraphs". 541 prompts, 25 instruction types.
- **Example:** "Make it more exciting. Please expand it into exactly 2 paragraphs, separated by `***`."
- **Scoring:** a program checks each instruction. Four numbers are reported (per prompt or per instruction, strict or loose), and labs don't all report the same one.
- **Human baseline:** none reported.
- **Top score:** 95.0% (Qwen3.5-27B, self-reported). Near its ceiling since about 2024-12 (Llama 3.3 70B: 92.1%).
- **Qwen3.5-4B:** 1 of 3 off, 3 of 3 on. Qwen's card lists 89.8% (mode not stated).

## RULER (2024-04, NVIDIA)
- **Tests:** whether a model can really use a long context: find a hidden "needle" in a long text, trace variables, count, answer questions. The text is made by a program, at lengths from 4,000 to 128,000 tokens.
- **Example:** "What is the special magic number for lantern mentioned in the text above?", with the number hidden somewhere in a long text. Another task: "Find every variable that is assigned the value 96223, directly or through other variables."
- **Scoring:** recall-based accuracy per task, averaged; "effective length" is the longest length where the score stays above 85.6%.
- **Human baseline:** none (synthetic).
- **Top score:** about 96% average (Jamba-1.5-large, author-reported). Near its ceiling at the standard 128K setting; Gemini 1.5 Pro already got 95.8% in the original paper.
- **Qwen3.5-4B:** 3 of 3 both ways, but my questions used short contexts, so this says nothing about the long-context limit.
