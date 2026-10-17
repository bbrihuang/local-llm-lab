# Tool-calling and agent benchmark cards

One card per benchmark: what it tests, an example, how it is scored, the top score and whether it is maxed out, what the score depends on, and how Qwen3.5-4B did if its model card lists it. "Self-reported" means the model's maker published the number and nobody independent has checked it. Aggregator sites (BenchLM, BenchmarkList and similar) copy numbers from others, so their "top score" rows are unconfirmed. Sources and the full notes: [benchmarks.csv](data/benchmarks.csv), [sources.md](data/sources.md), model-report numbers in [report_scores.csv](data/report_scores.csv).

## BFCL v4, Berkeley Function Calling Leaderboard (v1 in 2024, v4 is current; UC Berkeley)
- **Tests:** whether a model calls functions (tools) correctly. v1 added AST checking, v2 added enterprise and open-source functions, v3 added multi-turn conversations, v4 adds "agentic" tests: web search, memory, and sensitivity to prompt format.
- **Example:** v4 web search: a human-written multi-hop question; the model must use a DuckDuckGo search tool and a page-fetch tool (with random fetch errors like 503 or 429 injected) and give a short exact answer.
- **Scoring:** single calls are checked by AST matching (does the call have the right function and arguments). Multi-turn checks the backend state after each turn plus the path of calls. Overall Accuracy is an average over sub-categories; one blog gives weights (Agentic 40%, Multi-Turn 30%, others 10% each) while the leaderboard page says unweighted, so I don't know which is current.
- **Size:** v4 agentic has 665 entries (200 web search, 465 memory) plus 5,200 format-sensitivity entries; v3 multi-turn has 800.
- **Human baseline:** none reported.
- **Top score:** unclear. One aggregator lists Claude Opus 4.5 (FC) 77.47%, another lists a model called BTL-3 at 88.5% (self-reported, unverified). **Not maxed out**: multi-turn is still hard.
- **Depends on:** native function-calling mode vs plain-text "Prompt" mode, the BFCL code version (leaderboard cites commit f7cf735), the search tool, and the memory backends. 23 of 24 rows on BenchLM are provider self-reports. No contamination note on the leaderboard page.
- **Qwen3.5-4B:** **50.3** on BFCL-V4 (self-reported; Qwen3.5-9B 66.1, Qwen3.5-397B-A17B 72.9). Settings not stated.

## tau-bench, tau2-bench, tau3-bench (2024-06, 2025-06, 2026; Sierra)
- **Tests:** a customer-service agent chats with a simulated user (played by an LLM), calls tools (APIs) and has to follow a written policy. tau-bench has retail (115 tasks) and airline (50 tasks). tau2 adds telecom and "dual control", where the user also takes actions (like toggling airplane mode). tau3 adds banking with a knowledge base and voice.
- **Example:** a retail user wants to exchange an item; the agent must check the order, follow the exchange policy, and edit the database correctly.
- **Scoring:** success means the final database matches the goal state. pass^k is the chance that all k repeats of a task succeed, so it measures consistency. pass^1 is the usual headline.
- **Human baseline:** none reported.
- **Top score:** on taubench.com, text tau2: Qwen3.5-397B-A17B 87.9% pass^1. A BenchLM display-only table shows models near 99% on the telecom domain from Artificial Analysis runs (aggregator, mixed setups). **Near its ceiling** on telecom; unconfirmed elsewhere.
- **Known problems:** an audit found errors in 25 of 50 original airline tasks (policy conflicts, ambiguous instructions, database or grading errors); excluding them doubled average pass^5 from 20.8% to 40.0%. Sierra fixed 50+ tasks in 2026-02 and 75+ more for tau3, so old and new scores are not comparable.
- **Depends on:** the user-simulator model (original paper: gpt-4-0613; the repo example uses gpt-4.1; Qwen3.5 and GLM-5 don't say), the task version, which domains are averaged, and prompt tweaks. Qwen3.5 and GLM-5 both use the airline fixes from the Claude Opus 4.5 system card, and GLM-5 added a prompt tweak for retail and telecom. No contamination study found.
- **Qwen3.5-4B:** **79.9** on TAU2-Bench (self-reported; Qwen3.5-9B 79.1). Surprisingly higher than the 9B; treat small gaps as noise.

## GAIA (2023-11, Meta, Hugging Face and AutoGPT people; affiliations unconfirmed)
- **Tests:** real-world assistant questions that need several steps: browsing, reading attached files (PDF, image, audio), reasoning and tool use. Easy for people, hard for AI. Three levels.
- **Example:** a question with an attached spreadsheet or image whose answer needs a web lookup and a calculation, and has one short correct answer.
- **Scoring:** exact match on the final short answer, overall and per level.
- **Size:** 466 questions (300 with hidden answers for the test leaderboard); 165 are in the validation set.
- **Human baseline:** 92% (paper). GPT-4 with plugins got 15% at launch.
- **Top score:** 74.55% on the validation set (Claude Sonnet 4.5 in the HAL Generalist scaffold, verified by HAL, late 2025). A secondary article says H2O.ai passed 75% on the test set (unconfirmed). **Near the ceiling** on Level 1 (82%); overall not fully maxed.
- **Depends on:** the scaffold and tools more than the model. A bare model scored 44.8 on one aggregator while a scaffolded agent got 74.55. Validation vs test set matters. The dataset is gated so nobody re-shares it, but it has been public since 2023, so leakage is possible (no study found).
- **Qwen3.5-4B:** not listed. GAIA is not on any of the four model reports I read.

## BrowseComp (2025-04, OpenAI)
- **Tests:** whether a browsing agent can find a hard-to-find fact on the web. Questions were built backwards from a known answer, so they are hard to search but easy to check.
- **Example:** a long clue-filled question (event with several constraints) whose answer is one short name; OpenAI's own page example answers "Plastic Man".
- **Scoring:** accuracy against a short reference answer, graded by an LLM. 1,266 questions.
- **Human baseline:** trainers solved 29.2% of the problems they tried in 2 hours (367 of 1,255); when they answered, 86.4% matched.
- **Top score:** 92.5% (a model called Atria Dawn Preview, BenchLM, all 57 rows self-reported, unverified). **Near its ceiling**: top three within 0.8 points. At launch GPT-4o got 0.6% and Deep Research 51.5%.
- **Contamination:** real. Anthropic found answers to 9 of 1,266 problems in public web content (papers, including ICLR 2026 submissions) and 2 runs where Claude Opus 4.6 realized it was being tested and decrypted the answer key. Adjusted score 86.57 vs 86.81.
- **Depends on:** the search and browser tools, context management (GLM-5: 62.0 discard-all vs 75.9 with its own strategy; Qwen3.5-397B: 69.0 vs 78.6), search budget, number of attempts or agents, and the grader model (GLM-5 used o3-mini). BrowseComp-Plus uses a fixed corpus and is not comparable.
- **Qwen3.5-4B:** not listed. For scale: DeepSeek-V4-Pro Max 83.4, GLM-5 75.9 (with context management), Qwen3.5-397B 78.6 (all self-reported).

## MCP-Universe (2025-08, Salesforce AI Research)
- **Tests:** agents using real MCP servers (MCP, the Model Context Protocol, is a standard plug for giving models tools) in 6 domains: location navigation, repository management, financial analysis, 3D design, browser automation, web search. 11 servers, 231 tasks.
- **Example:** a task using the GitHub server to manage a repository, or the Google Maps server to plan a route.
- **Scoring:** success rate with three evaluators: format, static (fixed answers), and dynamic (answers fetched live for time-sensitive questions).
- **Human baseline:** none reported.
- **Top score:** 44.59% (Gemini-3-Pro-Preview, function-call track, project page; date not shown). **Not maxed out.** At launch GPT-5 got 43.72%.
- **Depends on:** the agent loop (ReAct 43.72 vs function calls 41.99 for GPT-5-Medium), the servers and tool set, long context, and live data and API keys. No contamination discussion found.
- **Qwen3.5-4B:** not listed. Do not mix it up with MCP-Mark (on the Qwen3.5-397B card, 46.1) or MCPAtlas (on the DeepSeek-V4 and GLM-5 reports), which are different benchmarks.

## MCPBench (2025-04, ModelScope / Alibaba)
- **Tests:** not models. It compares MCP servers (for example Bing, Brave and DuckDuckGo search servers) by keeping the same model and agent fixed. Tasks: web search, database query, GAIA questions.
- **Scoring:** task accuracy, latency and token use. Best result in the report: Bing Web Search server 64% accuracy.
- **Human baseline:** not applicable. **Maxed out?** Not applicable, it is a server comparison, not a leaderboard. Task counts unconfirmed.
- **Depends on:** the fixed model and agent, the search backend, and network speed.
- **Qwen3.5-4B:** not applicable.

## Terminal-Bench 2.0 (2025-11, Stanford and the Laude Institute)
Already researched in Task 9. Full card: [../09-coding-benchmarks/cards.md](../09-coding-benchmarks/cards.md) (section "Terminal-Bench 2.0"), row in [../09-coding-benchmarks/data/benchmarks.csv](../09-coding-benchmarks/data/benchmarks.csv). Short version:
- **Tests:** an agent finishes hard tasks in a Docker terminal. 89 tasks. Scored by the share of tasks where every test passes.
- **Top score:** 84.7% on the Terminal-Bench leaderboard (unconfirmed whether checked); independent best 73.2% (Vals AI). **Not maxed out**, though the authors' under-50% goal no longer holds.
- **Depends on:** the agent harness. GLM-5: 56.2 with Terminus-2 and 56.2 with Claude Code; 60.7 and 61.1 on the verified version that fixes ambiguous instructions.
- **Qwen3.5-4B:** not listed. Qwen3.5-397B 52.5, DeepSeek-V4-Pro Max 67.9 (self-reported).
