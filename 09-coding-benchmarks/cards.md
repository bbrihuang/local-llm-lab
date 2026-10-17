# Coding benchmark cards

One card per benchmark: what it tests, an example, how it is scored, the top score and whether it is maxed out, and how Qwen3.5-4B (Q4_K_M, my RTX 5070) did. "Self-reported" means the model's maker published the number and nobody independent has checked it. Sources and the full notes: [data/benchmarks.csv](data/benchmarks.csv), [data/sources.md](data/sources.md).

## HumanEval (2021-07, OpenAI)
- **Tests:** writing one Python function from a docstring. 164 problems.
- **Example:** a function header such as `def has_close_elements(numbers, threshold)` with a docstring saying to check whether any two numbers are closer than the threshold. The model writes the body.
- **Scoring:** the function must pass hidden unit tests. pass@1 (pass on the first try) is the usual headline.
- **Human baseline:** none reported.
- **Top score:** about 96% (listed on an aggregator, vendor-reported). **Near its ceiling:** GPT-4 got 67.0% in 2023-03, and 90%+ scores appeared by 2024-05. The tests are weak (EvalPlus added about 80x more tests and scores dropped by up to 19 to 29 points) and the problems are widely copied online, so contamination is likely.
- **Qwen3.5-4B (mine):** **83.5%** thinking off, **80.5%** thinking on (164 problems, plus or minus 3). See the README.

## MBPP (2021-08, Google)
- **Tests:** entry-level Python problems written by crowd workers from a one-line description. 974 problems.
- **Example:** a sentence like "write a function to find the shared elements from two lists", with 3 assert tests that the program must pass.
- **Scoring:** pass@k on the assert tests, often with 3 examples in the prompt.
- **Human baseline:** none reported.
- **Top score:** about 94.9% (o4-mini, from an aggregator site, undated; unconfirmed). **Near its ceiling** (date unconfirmed). EvalPlus found broken tasks and wrong reference solutions and made a cleaned version (MBPP+).
- **Qwen3.5-4B:** not run. For comparison, Qwen3-4B-Base got 67.0 (3-shot, self-reported).

## LiveCodeBench (2024-03, UC Berkeley, MIT and Cornell; unconfirmed)
- **Tests:** competition programming problems (LeetCode, AtCoder, CodeForces) collected continuously, so you can score a model only on problems released after its training cutoff. This is how it avoids leaked test data.
- **Example:** a contest-style problem statement with an input and output format; the model writes a full program checked by hidden tests.
- **Scoring:** pass@1 on hidden tests, filtered by problem release date. Versions (v5: 880 problems, v6: 1,055) and date windows differ between reports, so scores from different papers aren't comparable.
- **Human baseline:** none reported.
- **Top score:** 93.5% (DeepSeek-V4-Pro Max, self-reported, version not stated). **Near its ceiling** since about 2026-04.
- **Qwen3.5-4B:** not run. Its model card lists **55.8** on LiveCodeBench v6 (self-reported).

## SciCode (2024-07)
- **Tests:** real scientific-computing code from research in physics, math, chemistry, biology and materials science. 80 main problems split into 338 dependent subproblems, with scientist-written solutions.
- **Example:** a multi-step physics problem where step 3 needs your step 1 and 2 functions.
- **Scoring:** the fraction of subproblems solved, and the stricter main-problem accuracy, which needs every subproblem right.
- **Human baseline:** none reported.
- **Top score:** 66.9% on subproblems (Claude Opus 5.5, independent run by Artificial Analysis, with an unconfirmed metric definition). **Not maxed out.** A 2026 audit (SciCode-Verified, author-reported) found test defects that reject correct solutions, so public scores are probably too low.
- **Qwen3.5-4B:** not run; not listed on its card.

## SWE-bench Verified (2024-08, OpenAI with the Princeton SWE-bench authors)
- **Tests:** fixing real GitHub issues in 12 popular Python projects. 500 problems, picked by human reviewers from 2,294.
- **Example:** my task, `psf__requests-1142`: the `requests` library sends `Content-Length: 0` on every GET request, and some servers reject it. The fix is about three lines.
- **Scoring:** the "resolved" rate: the tests that failed must now pass, and the ones that passed must still pass. The model runs inside a scaffold (an agent loop with tools), and the scaffold matters a lot, so scores aren't comparable across vendors.
- **Human baseline:** none reported.
- **Top score:** 97.0% (Claude Opus 5, independent run by Vals AI, 2026-09, with a bash-only agent). **Maxed out** (Vals marks it saturated; the first 80%+ was Claude Opus 4.5 at 80.9% in 2025-11). In 2026-02 OpenAI said 59.4% of the 138 hardest problems it audited had flawed tests or descriptions, found that frontier models could reproduce the real fixes, and stopped reporting it.
- **Qwen3.5-4B:** not run; not on its card. For scale, the open models in the reports: DeepSeek-V4-Pro Max 80.6, GLM-5 77.8, Qwen3.5-397B 76.4 (all self-reported).

## Terminal-Bench 2.0 (2025-11, Stanford and the Laude Institute)
- **Tests:** an agent completes hard tasks in a Docker terminal: software engineering, system administration, security and more. 89 tasks in 16 categories.
- **Example:** an instruction such as setting up or repairing a service in a container; tests then check the container's final state.
- **Scoring:** the share of tasks where every test passes; results depend on the agent harness and the model.
- **Human baseline:** none reported (each task has a human-written reference solution).
- **Top score:** 84.7% on the Terminal-Bench leaderboard (an agent with GPT-5.5; whether the maintainers checked the run is unconfirmed). The best independent run I found is 73.2% (Vals AI, 2026-06). **Not maxed out**, but the authors' goal of keeping frontier models under 50% no longer holds, and a successor (Terminal-Bench 4.0) exists. 43 of 45 rows on one aggregator are provider self-reports.
- **Qwen3.5-4B:** not run; not on its card. Reported for others: DeepSeek-V4-Pro Max 67.9, GLM-5 56.2, Qwen3.5-397B 52.5 (self-reported).
