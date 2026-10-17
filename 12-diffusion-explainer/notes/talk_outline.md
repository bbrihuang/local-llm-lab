# 30-minute talk: how diffusion works (before Task 11's demo)

Audience: the group, comfortable with LLMs, not necessarily with diffusion. Centre of the talk: the 2D animation. Draft outline; slide count about 18.

| Min | Slides | Content |
|---|---|---|
| 0-3 | 1-2 | **Hook.** Two ways to write text: one word at a time (what they know) or the whole thing at once, then refined. Show the finished Task 11 demo for 20 seconds and promise to explain it. |
| 3-8 | 3-5 | **Diffusion in one picture.** Destroy data with noise step by step; learn to undo one step; generate by starting from noise. Use an image example. |
| 8-17 | 6-10 | **Flow matching, live.** Straight path between a noise point and a data point; the network predicts the direction; sampling is following arrows. Play `flow_animation.gif`. Then `flow_steps.png`: one step gives the blurry average, 2 and 8 steps sharpen it (numbers: 0.378, 0.157, 0.036 against 0.026 at 100 steps). Why the loss never reaches zero (average of many paths). |
| 17-22 | 11-13 | **Text needs a discrete version.** No half way between cat and dog. Replace noise with masks. Masked diffusion training (BERT with a random mask rate) and generation (start all masked, unmask a few per round). One slide each for MDLM, LLaDA and SEDD (ratios instead of a score). |
| 22-26 | 14-15 | **Block diffusion = DiffusionGemma.** Autoregressive across blocks, diffusion inside a block of 256 tokens. Why: variable length and cache reuse. |
| 26-29 | 16-17 | **Diffusion LM against autoregressive.** The comparison table: parallel, fewer steps on easy text, quality gap. What Task 11 will measure. |
| 29-30 | 18 | **Questions, and the bridge to the demo.** |

Prepare: the two PNGs and the GIF from `results/`, a masked-text example (a sentence with 0%, 50%, 100% of words masked), the block diagram of block diffusion.

Note: slides not built yet (deferred, like the other presentations). Verify the paper claims in `explainer.md` first.
