# Task 4 readings: different ways of quantizing

Notes on 8 sources, written for a beginner. Every number below was read in the source unless it is marked **not verified**. Table and figure numbers refer to the paper named in that section.

Key words used throughout:
- **Weights**: the numbers the model learned during training. They are fixed after training.
- **Activations**: the numbers flowing through the model while it runs. They change with every input.
- **Perplexity (PPL)**: how "surprised" the model is by real text. Lower is better. WikiText-2 and C4 are standard text sets used to measure it.
- **Zero-shot accuracy**: the score on benchmark tasks the model was not tuned for. Higher is better.
- **W4A16, W8A8, W4A4**: shorthand for weight bits and activation bits. W8A8 means 8-bit weights and 8-bit activations.
- **PTQ (post-training quantization)**: quantize a model after it is trained. **QAT (quantization-aware training)**: train with quantization in the loop.
- **Calibration data**: a small batch of example text that is run through the model to measure value ranges or errors before quantizing.

---

## Naive quantization and why it fails (from the Visual Guide)

**Why quantize.** A 70B-parameter model stored in 32-bit floats needs **280 GB** just to load. FP16 and BF16 use 16 bits per value, which halves that. INT8 uses 8 bits.

**Absmax (symmetric) quantization.** Find the largest absolute value α in the data. Map the range [-α, α] onto the integers [-127, 127]:
`x_q = round(127 · x / α)`. To dequantize, compute `x ≈ x_q · α / 127`. Zero stays exactly zero. This is Eq. 0 in LLM.int8(). The Visual Guide shows its formulas as images. Its example: **3.08 and 3.02 both become 36**, so after dequantizing you can no longer tell them apart. That loss is the **quantization error**.

**Zero-point (asymmetric) quantization.** Map the actual minimum β and maximum α (the Guide's example range is [-7.59, 10.8]) onto [-128, 127]. Because the range is not centered on zero, you also store a **zero-point** z, the integer that real 0 maps to. This makes better use of the levels when data is lopsided, but it costs extra math.

**Outliers break it.** An **outlier** is a value far larger than the rest. With absmax, a single outlier sets α. Every normal value gets squeezed into a few integer levels, so their differences are lost. **Clipping** caps the range (the Guide uses [-5, 5]). This helps the normal values but badly distorts the outliers.

**Granularity: how many values share one scale.**
- **Per-tensor**: one scale for a whole matrix. It is cheapest, and one outlier ruins the whole matrix.
- **Per-channel / per-row / per-token**: one scale per row or column. Outliers are contained to their own row or column.
- **Group-wise / block-wise**: one scale per small block, for example 128 weights. GGUF (llama.cpp's format) uses "super-blocks" that contain "sub-blocks." Each sub-block uses absmax, and its scale is itself quantized using the super-block's scale. Smaller groups give better accuracy, at the cost of more stored scales.

(The Guide does not name per-tensor or per-channel explicitly. Those terms come from LLM.int8(), SmoothQuant and GPTQ below.)

**Static vs dynamic.** Weights can be calibrated once, ahead of time. Activations need either **dynamic** quantization (compute the scale live during inference, which is more accurate but slower) or **static** quantization (fix the scale from calibration data, which is faster but less accurate).

**A concrete failure (LLM.int8(), Table 1, C4 perplexity, 13B model):** FP32 gives **12.45**. Naive per-tensor Int8 absmax gives **19.08**. Int8 zero-point gives 13.94. Even per-row/column ("vector-wise") absmax gives 16.48. Naive 8-bit gets *worse* as models grow. The papers below are all ways around this.

---

## 1. LLM.int8() (Dettmers, Lewis, Belkada, Zettlemoyer, 2022)

**Problem it solves.** Before this paper, 8-bit methods lost accuracy and had only been tested up to about 350M parameters. The authors found that big models develop **outlier features**: a few hidden dimensions with values up to 20x larger than the rest. At about **6.7B parameters** a "phase shift" happens. These outliers then appear in all layers and about 75% of token positions. At 6.7B there are roughly 150,000 outliers per 2,048-token sequence, but they sit in only **6 feature dimensions**. Zeroing those dimensions raises perplexity by 600-1000%. Zeroing 7 random dimensions raises it by only about 0.1% (Section 4.2).

**Core idea.** Do two things together:
1. **Vector-wise quantization.** Give every row of the activations and every column of the weights its own scale.
2. **Mixed-precision decomposition.** Any hidden dimension with a value above **6.0** is treated as an outlier. Those few columns are multiplied in 16-bit, and everything else in Int8. The two results are then added.

Key formula (Eq. 8): `C ≈ Σ_{h∈O} X_f16^h W_f16^h + S · Σ_{h∉O} X_i8^h W_i8^h`, where O is the set of outlier dimensions.

**What gets quantized.** Weights and activations go to Int8 in the feed-forward and attention projection layers. More than **99.9%** of the values are multiplied in 8-bit (abstract). Outlier columns stay FP16. Outliers are detected live, so no calibration data is needed. No retraining is needed: you load a 16/32-bit checkpoint, convert it, and use it immediately.

**Headline results.**
- Table 1 (C4 PPL): absmax LLM.int8() matches FP32 at every size. Examples: 6.7B scores 13.24 vs 13.30, and 13B scores 12.45 vs 12.45. All naive 8-bit baselines degrade.
- Figure 1: on OPT models from 125M to 175B, zero-shot accuracy "maintains full 16-bit performance." The plain 8-bit baseline collapses to random. The text gives no exact numbers.
- BLOOM-176B memory drops **1.96x**. Large 175B-sized matrix multiplications run about **2x faster** (Section 3.4). The abstract says this makes OPT-175B and BLOOM usable on a single server with consumer GPUs.

**Cost and limits.** The overhead slows inference for models **below 6.7B** compared with FP16. Mixed precision is awkward for hardware. SmoothQuant measured LLM.int8() as *slower* than FP16 (OPT-30B: 387.9 ms vs 343.0 ms, SmoothQuant Table 11). The perplexity study covers models only up to 13B. The authors expected FP8 to beat Int8 but had no hardware for it.

**How it improves on what came before.** It was the first method to show that 8-bit inference at 175B scale can be lossless, by isolating the few outlier dimensions instead of letting them wreck the scale for everyone.

---

## 2. SmoothQuant (Xiao, Lin, Seznec, Wu, Demouth, Han, 2022)

**Problem it solves.** Weights are easy to quantize because they are flat and evenly spread. Activations are hard because of the outlier channels. Per-channel activation scales would fix this, but INT8 matrix-multiply hardware only supports scales per token (row) or per output channel. LLM.int8() keeps accuracy but is slow. ZeroQuant fails on OPT-175B.

**Core idea.** Move the difficulty from activations to weights with a math trick that does not change the output. Divide each activation channel by a factor s, and multiply the matching weight row by the same s. Activation outliers shrink, and weights get slightly spikier, which they can tolerate. This is done offline and folded into the previous layer, so it costs nothing at runtime.

Key formulas: `Y = (X·diag(s)⁻¹)·(diag(s)·W)` (Eq. 3), with `s_j = max|X_j|^α / max|W_j|^(1−α)` (Eq. 4). α is the **migration strength**. α = 0.5 works for OPT and BLOOM. GLM-130B needs 0.75. The best range is 0.4-0.6 (Fig. 10).

**What gets quantized.** Weights and activations, **W8A8**, for all linear layers and attention matrix multiplications. LayerNorm and Softmax stay FP16. There are three speed levels: O1 (per-token dynamic activations), O2 (per-tensor dynamic) and O3 (per-tensor static, the fastest). Calibration uses **512 random sentences from the Pile**, done once. No retraining.

**Headline results.**
- OPT-175B (Table 3), zero-shot average / WikiText PPL:

  | Method | Accuracy | PPL |
  |---|---|---|
  | FP16 | 66.9% | 10.99 |
  | Naive W8A8 | 35.5% | 93,080 |
  | ZeroQuant | 35.8% | 84,648 |
  | LLM.int8() | 66.7% | 11.10 |
  | SmoothQuant-O3 | 66.8% | 11.17 |
- Llama-2-70B WikiText-2 PPL is 3.320 → 3.359. Mixtral-8x7B is 3.842 → 3.893 (Table 7).
- Up to **1.56x** faster and nearly half the memory (FasterTransformer, Fig. 9). OPT-175B runs on 4 GPUs instead of 8. **MT-NLG 530B** is served on one 8-GPU node instead of 16 GPUs, with memory going from about 1,040 GB to 527 GB (Tables 9-10).

**Cost and limits.** α must be tuned per model. Static O3 loses some accuracy on some models: BLOOM about 0.8%, and GLM-130B about 1%, which also needed clipping during calibration. It is designed for 8-bit. At 4 bits it breaks: QuaRot's Table 1 shows SmoothQuant at **83.12 PPL** on Llama-2-7B at W4A4.

**How it improves on what came before.** It gets LLM.int8()-level accuracy with plain, uniform INT8 kernels, so it is actually *faster* than FP16 instead of slower.

---

## 3. GPTQ (Frantar, Ashkboos, Hoefler, Alistarh, 2022)

**Problem it solves.** Round-to-nearest (RTN, the naive method) works at 8 bits but falls apart at 3-4 bits. Accurate methods existed, but they were far too slow for 100B+ models. For example, ZeroQuant-LKD takes about 3 hours for a 1.3B model.

**Core idea.** Quantize the weights one column at a time. After each column is rounded, slightly adjust the *not-yet-quantized* weights to cancel out the error just made. The goal is that the layer's **output** stays the same, not that each weight stays close to its original value. To know how to adjust, GPTQ uses the **Hessian**, a matrix of second derivatives that says how sensitive the output error is to each weight and to pairs of weights. For a linear layer it is simply built from the calibration inputs: `H = 2·X·Xᵀ`. GPTQ is a scaled-up version of an older method, OBQ (Optimal Brain Quantization). Three tricks make it fast:
1. Quantize all rows in the **same column order**, so one inverse Hessian is shared.
2. **Lazy batch updates** in blocks of 128 columns, which use the GPU efficiently.
3. A **Cholesky** decomposition (a stable way to factor a matrix) to avoid numerical blow-ups.

Key formula (the OBQ update GPTQ inherits; not quoted from the fetched text, **not verified** as written): `δ = −(w_q − quant(w_q)) / [H⁻¹]_qq · (H⁻¹)_:,q`. In words, the rounding error of weight q is spread over the remaining weights in proportion to the inverse Hessian.

**What gets quantized.** **Weights only**, at 3 or 4 bits, sometimes with groups of 1024 or 128 (about 0.02 or 0.15 extra bits per weight). Activations stay FP16. Calibration uses **128 random 2,048-token segments from C4**. No retraining. Quantizing OPT-175B takes **4.2 hours** and BLOOM-176B **3.8 hours** on one A100 (Table 2).

**Headline results** (WikiText-2 PPL):

| Model | FP16 | RTN 4-bit | GPTQ 4-bit | RTN 3-bit | GPTQ 3-bit |
|---|---|---|---|---|---|
| OPT-175B (Table 3) | 8.34 | 10.54 | **8.37** | 7.3e3 | **8.68** |
| BLOOM-176B (Table 4) | 8.11 | 8.37 | **8.21** | 571 | **8.64** |

- 4-bit RTN on OPT-175B (10.54) is worse than the *unquantized 13B model* (10.13). GPTQ loses only 0.03.
- 3-bit OPT-175B is about 63 GB and fits on **one 80 GB A100**. FP16 needs 5 of them.
- End-to-end generation is about **3.25x** faster on A100 and **4.5x** on A6000.

**Cost and limits.** It needs calibration data and a few GPU-hours. The speedup comes only from moving less memory, because hardware lacks FP16×INT4 math. It does not quantize activations. OPT-66B is an odd exception that quantizes worse. AWQ argues it can **overfit the calibration set**.

**How it improves on what came before.** It made error-correcting ("second-order") quantization fast enough for 175B models, pushing usable weight quantization from 8 bits down to 3-4 bits.

---

## 4. AWQ: Activation-aware Weight Quantization (Lin, Tang, Tang, Yang, Dang, Gan, Han, 2023)

**Problem it solves.** Not all weights matter equally. Low-bit RTN hurts the important ones. Keeping them in FP16 (mixed precision) is unfriendly to hardware. GPTQ's error reconstruction can overfit to the calibration text.

**Core idea.** About **0.1-1% of weights are "salient"** (important). The surprise is that you find them by looking at the **activations**, not the weights: a weight channel matters if the inputs flowing into it are large. Instead of keeping those channels in FP16, AWQ **scales them up** before quantizing (and scales the input down by the same amount). This shrinks their *relative* rounding error. There is no backpropagation and no reconstruction, only a small grid search.

Key formula: `s = s_X^α`, where s_X is the average activation size for each input channel. α ∈ [0, 1] is chosen by grid search (20 points) to minimize the layer's output error. Weight clipping is then tuned as well.

**What gets quantized.** **Weights only**, INT4 or INT3, in groups of 128 (W4A16). It uses a "small calibration set" from the Pile. The exact size was not in the text I read; 128 samples is **not verified**. No retraining.

**Headline results.**
- Table 1 (OPT-6.7B, INT3-g128, PPL; FP16 is 10.86): RTN gives 23.54. Keeping 1% of channels in FP16, *chosen by activation*, gives **11.39**. Chosen by weight size gives 22.37, and chosen randomly gives 24.23.
- Table 2: scaling those channels by s = 2 gives 11.92, the best value. s = 4 is worse (12.36) because it starts to hurt the other channels.
- Table 4 (WikiText-2 PPL, g128):

  | Model, bits | FP16 | RTN | GPTQ | AWQ |
  |---|---|---|---|---|
  | Llama-2-7B, INT3 | 5.47 | 6.66 | 6.43 | **6.24** |
  | Llama-2-70B, INT3 | 3.32 | 3.98 | 3.88 | **3.74** |
  | Llama-2-7B, INT4 | 5.47 | 5.73 | 5.69 | **5.60** |
- It works on Mistral and Mixtral MoE (Table 5) and on multimodal OpenFlamingo (Table 6: INT4 COCO drop of −1.17, vs −4.57 for RTN and −6.72 for GPTQ).
- The TinyChat runtime is **more than 3x faster** than Hugging Face FP16 on desktop and mobile GPUs. Llama-2-70B runs on a single Jetson Orin (64 GB).

**Cost and limits.** It is weights only, so the speedup comes from memory traffic during token generation, not from cheaper math. Activations stay 16-bit. There is still a gap to FP16 at INT3 (LLaMA-7B: 6.35 vs 5.68). The paper claims better robustness to the calibration set (Section 5.3, Figure 8), but I could not read those numbers.

**How it improves on what came before.** It gets GPTQ-or-better 3-4-bit accuracy with a simpler, data-light, hardware-friendly trick (a per-channel scale), and it generalizes better across domains and model types.

---

## 5. QuaRot (Ashkboos, Mohtashami, Croci, Li, Cameron, Jaggi, Alistarh, Hoefler, Hensman, 2024)

**Problem it solves.** Fast inference needs *all* matrix multiplications in 4 bits: weights, activations and the **KV cache** (the stored keys and values from earlier tokens). At 4 bits, activation outliers are deadly. Earlier methods either kept outlier channels in higher precision (QUIK, Atom) or needed retraining (OmniQuant).

**Core idea.** **Rotate** the hidden state so that outliers are smeared evenly across all dimensions. Then there is nothing extreme left to clip. This works because of **computational invariance**: if you multiply a layer's input weights by an **orthogonal matrix** Q (a rotation, with `Q·Qᵀ = I`) and the next layer's weights by Qᵀ, the model's output is unchanged. Most rotations are folded into the weights for free. A few are done live ("online").

QuaRot uses **randomized Hadamard matrices**. A **Hadamard matrix** is a square matrix of only +1 and −1 entries (scaled) whose rows are perpendicular to each other. It can be applied in `O(d·log d)` operations, like a fast Fourier transform. Built as `H_{2^n} = H_2 ⊗ H_{2^{n−1}}`, with `H_2 = (1/√2)[[1,1],[1,−1]]`, then multiplied by random ±1 signs.

**What gets quantized.** Everything at INT4: **W4A4KV4**. Activations are quantized per token (clip 0.9), the KV cache in groups of 128, and weights per column. Weights use GPTQ by default, which needs **128 WikiText-2 samples of 2,048 tokens**, about 2 hours for 70B on one A100. Alternatively they use RTN, which needs **no calibration** but is less accurate at 4 bits. No retraining.

**Headline results.**
- Table 1 (WikiText-2 PPL, W4A4):

  | Method | Outliers kept | Llama-2-7B | 13B | 70B |
  |---|---|---|---|---|
  | FP16 | – | 5.47 | 4.88 | 3.32 |
  | SmoothQuant | 0 | 83.12 | 35.88 | – |
  | OmniQuant | 0 | 14.26 | 12.30 | – |
  | QUIK-4B | 256 | 8.87 | 7.78 | 6.91 |
  | **QuaRot** | **0** | **6.10** | **5.40** | **3.79** |
- 70B loses only **0.47** PPL. Table 2 zero-shot average: 70B goes from 77.07 to 75.98, about **99% retained**. Note that 7B loses more, 69.82 → 65.64.
- On a single transformer block on an RTX 3090: up to **3.33x** prefill speedup and **3.89x** decoding memory savings for 70B.
- At 6 and 8 bits, plain RTN with no calibration is described as lossless.

**Cost and limits.** Online Hadamard transforms add runtime work. The best accuracy still needs GPTQ plus calibration data. Speedups were measured on one transformer block, not the full model. Smaller models lose more accuracy at W4A4.

**How it improves on what came before.** Instead of *protecting* outliers (LLM.int8(), QUIK) or *shifting* them (SmoothQuant), it *removes* them with rotations. That made true end-to-end 4-bit inference work with zero outlier channels.

---

## 6. Microscaling FP4 / MR-GPTQ (Egiazarian, Castro, Kuznedelev, Panferov, Kurtić, Pandit, Marques, Kurtz, Ashkboos, Hoefler, Alistarh, 2025)

**Background terms.**
- **FP4 E2M1**: a 4-bit floating-point number with 1 sign bit, 2 exponent bits and 1 mantissa bit. It can only represent ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. Unlike INT4, its levels are bunched near zero.
- **Microscaling**: a block of values shares one small scale factor, so this is group-wise quantization built into the hardware format.
  - **MXFP4**: groups of 32 with an **E8M0** scale, an 8-bit power-of-two-only number. About 4.25 bits per value.
  - **NVFP4**: groups of 16 with an **FP8 E4M3** scale (finer-grained). About 4.5 bits per value.
- NVIDIA Blackwell GPUs (B200, RTX 5090) run both formats natively. The paper says AMD will support MXFP4.

**Problem it solves.** FP4 promises fast, accurate 4-bit inference, but the real-world results disappoint. MXFP4's power-of-two scales create large rounding error, about a 10% relative accuracy drop. With NVFP4's tiny groups of 16, the classic outlier fixes stop helping. The authors' math (Lemma 2, Figure 3) predicts, and experiments confirm, that **Hadamard rotation hurts at small group sizes** (NVFP4 with RTN) and helps at larger ones (MXFP4).

**Core idea.** **MR-GPTQ (Micro-Rotated GPTQ)** is GPTQ adapted to FP4. It has three parts:
1. Hadamard rotations applied **within each small block** rather than across the whole hidden size, fused into the weights, with a fast online rotation for activations.
2. Scale grids tuned to minimize error for each specific format.
3. Static activation reordering.

It ships with **QuTLASS** GPU kernels so the rotations add negligible overhead.

**What gets quantized.** Weights and activations, **W4A4**, in MXFP4 or NVFP4. GPTQ and SpinQuant baselines are calibrated on **1,024 FineWeb sequences** (Section 5.1). The calibration set for MR-GPTQ itself was not stated explicitly in what I read. This is post-training, with no retraining (a QAT comparison is also included).

**Headline results.**
- Table 1, Llama-3.1-8B-Instruct, simulated W4A4, % of FP16 accuracy recovered:

  | Method | NVFP4 | MXFP4 |
  |---|---|---|
  | RTN | 94.67 | 87.83 |
  | GPTQ | 95.92 | 89.47 |
  | QuaRot | 93.80 | 79.70 |

  INT4 (group 32): RTN 92.63, GPTQ 92.75.
- Table 2 / Fig. 5, real kernels, MXFP4 recovery RTN → MR-GPTQ: Llama-3 8B **88.1 → 93.3**, Llama-3 70B 96.8 → 98.4. NVFP4 8B is 94.8 → 95.8. MR-GPTQ brings MXFP4 to within 1-2% of NVFP4, and large models reach **98-99%** recovery.
- Speedups over FP16 are up to **3.6x layer-wise and 2.2x end-to-end on B200**, and **6x layer-wise and 4x end-to-end on RTX 5090**.

**Cost and limits.** "No lossless format": every FP4 format shows a noticeable drop. The ranking is NVFP4 > INT4 > MXFP4. Small models (≤ 8B) and Llama recover worse than Qwen3. The speed results require Blackwell GPUs. The authors' conclusion is that FP4 is *not automatically better than INT4*.

**How it improves on what came before.** It shows that QuaRot-style global rotations can *hurt* on new hardware formats. By making rotations and GPTQ block-aware, it makes the cheaper MXFP4 nearly as accurate as NVFP4.

---

## 7. BitNet b1.58: "The Era of 1-bit LLMs" (Ma, Wang, Ma, Wang, Wang, Huang, Dong, Wang, Xue, Wei, 2024)

**Problem it solves.** Post-training quantization always starts from a full-precision model and loses accuracy at very low bits. Big models are also limited by moving weights from memory and by energy use.

**Core idea.** **Train the model from scratch** with every weight restricted to **{−1, 0, +1}**. Three possible values carry log₂3 ≈ **1.58 bits** of information. Multiplying by −1, 0 or +1 is just subtract, skip or add, so matrix multiplication needs **no floating-point multiplications**. The 0 lets the model switch features off. All `nn.Linear` layers become **BitLinear**. The rest is LLaMA-style (RMSNorm, SwiGLU, rotary embeddings, no biases).

Key formula (**absmean** quantization): `γ = mean(|W|)`, then `W̃ = RoundClip(W / (γ + ε), −1, 1)`.

**What gets quantized.** Weights are ternary (1.58 bits). Activations are 8-bit, scaled per token. **Training from scratch is required**, so it cannot convert an existing model. No calibration step.

**Headline results.**
- Table 1, vs an FP16 LLaMA trained the same way (100B RedPajama tokens):

  | Size | PPL (LLaMA / BitNet) | Memory | Latency |
  |---|---|---|---|
  | 700M | 12.33 / 12.87 | 2.60x less | 1.23x faster |
  | 3B | 10.04 / **9.91** | 3.55x less | 2.71x faster |
- Table 2, zero-shot average at 3B: 49.7 (LLaMA) vs **50.2** (BitNet). It matches FP16 "starting from 3B" and is slightly behind at 700M and 1.3B.
- 70B is **4.1x** faster. Table 3: **8.9x** throughput (2,977 vs 333 tokens/s) and an 11x larger batch on two A100s.
- Estimated **71.4x** lower arithmetic energy for matrix multiplication (from an energy model, not measured).
- Table 4, trained on 2T tokens, 3B: BitNet averages 74.34 vs StableLM-3B's 73.22.

**Cost and limits.** It requires full pretraining, which is huge cost. Accuracy was tested only up to 3.9B. The 7B-70B numbers are speed, memory and energy projections, not quality. The StableLM numbers were copied from its report, not re-run. Real gains need custom kernels or hardware.

**How it improves on what came before.** It flips the question from "how do we compress a trained model?" to "train it low-bit from the start," and gets near-FP16 quality with weights below 2 bits.

---

## Comparison table

| Method | Year | What is quantized | Bits | Calibration? | Training? | Key trick | Main weakness |
|---|---|---|---|---|---|---|---|
| Naive RTN (absmax / zero-point) | – | W (and A) | 8 OK, ≤4 breaks | No | No | Round each value to the nearest level | Outliers wreck the scale |
| LLM.int8() | 2022 | W + A | 8 (outlier cols FP16) | No | No | Keep about 0.1% outlier dims in FP16 | Slower than FP16; 8-bit only |
| SmoothQuant | 2022 | W + A | W8A8 | Yes (512 Pile sentences) | No | Move outliers from A to W via scale s | Tune α; fails at 4-bit |
| GPTQ | 2022 | W only | 3-4 (W4A16) | Yes (128 C4 segments) | No | Hessian-based error compensation | GPU-hours; may overfit calib.; no A quant |
| AWQ | 2023 | W only | 3-4 (W4A16, g128) | Yes (small Pile set) | No | Scale salient channels found via activations | Weights only; INT3 gap remains |
| QuaRot | 2024 | W + A + KV cache | W4A4KV4 | Optional (GPTQ yes, RTN no) | No | Hadamard rotations erase outliers | Online transforms; small models lose more |
| MR-GPTQ (MXFP4/NVFP4) | 2025 | W + A | FP4 (E2M1) + block scales | Yes | No | Block-wise rotations + FP4-aware GPTQ | Not lossless; needs Blackwell |
| BitNet b1.58 | 2024 | W (ternary) + A (8-bit) | 1.58 / 8 | No | **Yes, from scratch** | Ternary weights, multiply-free | Must pretrain; tested ≤ 3.9B |

---

## Sources (URLs actually read)

1. Visual Guide to Quantization (Maarten Grootendorst): https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization
2. LLM.int8(): https://ar5iv.labs.arxiv.org/html/2208.07339 and https://arxiv.org/abs/2208.07339v2
3. SmoothQuant: https://arxiv.org/html/2211.10438v7
4. GPTQ: https://ar5iv.labs.arxiv.org/html/2210.17323
5. AWQ: https://arxiv.org/html/2306.00978v5 and https://ar5iv.labs.arxiv.org/html/2306.00978
6. QuaRot: https://arxiv.org/html/2404.00456v2
7. Microscaling FP4 / MR-GPTQ: https://arxiv.org/html/2509.23202v1, https://arxiv.org/abs/2509.23202v2, https://arxiv.org/pdf/2509.23202v2
8. BitNet b1.58: https://arxiv.org/html/2402.17764v1

Gaps: LLM.int8() Appendix D speed table and AWQ Section 5.3 / Figure 8 (calibration robustness) did not come through in the fetched text, so they are not quoted.
