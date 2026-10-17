# Task 4 results (Qwen3-0.6B, fake quantization), pasted from Brian's run 2026-10-17

Whole-model WikiText-2 test perplexity (40 x 2,048 tokens; 32 x 2,048 calibration):

| Method | Perplexity | vs fp |
|---|---|---|
| fp (bf16) | 19.060 | 0 |
| w4_rtn_tensor | 789,566 | huge |
| w4_rtn_channel | 32.830 | +72.2% |
| w4_rtn_g128 | 23.355 | +22.5% |
| w4_gptq_g128 | 22.057 | +15.7% |
| w4_awq_g128 | 21.316 | +11.8% |
| w4_hadamard_g128 | 27.784 | +45.8% |
| w4_mxfp4 | 23.119 | +21.3% |
| w4_nvfp4 | 21.351 | +12.0% |
| w1.58_ternary | 2,561,980 | huge |
| w8a8_tensor | 33.098 | +73.7% |
| w8a8_token | 19.349 | +1.5% |
| w8a8_llm_int8 | 19.058 | -0.0% |
| w8a8_smoothquant | 19.138 | +0.4% |
| w4a4_token | 93,157 | huge |
| w4a4_smoothquant | 2,864.5 | +14,929% |
| w4a4_hadamard | 89.441 | +369% |
| w4a4_hadamard_gptq | 34.492 | +81.0% |
| w4a4_mxfp4 | 30.142 | +58.1% |
| w4a4_hadamard_mxfp4 | 41.268 | +116.5% |
| w4a4_nvfp4 | 23.853 | +25.1% |
| w4a4_hadamard_nvfp4 | 34.351 | +80.2% |

Per-layer median relative output error (layers 2, 14, 26; q/o/up/down; 8 test x 512 tokens, 64 x 512 calibration):

| Method | Error |
|---|---|
| w4_rtn_tensor | 0.4741 |
| w4_rtn_channel | 0.1529 |
| w4_rtn_g128 | 0.1094 |
| w4_gptq_g128 | 0.0632 |
| w4_awq_g128 | 0.0885 |
| w4_hadamard_g128 | 0.1080 |
| w4_mxfp4 | 0.1089 |
| w4_nvfp4 | 0.0848 |
| w1.58_ternary | 0.5674 |
| w8a8_tensor | 0.1115 |
| w8a8_token | 0.0363 |
| w8a8_llm_int8 | 0.0080 |
| w8a8_smoothquant | 0.0192 |
| w4a4_token | 0.4246 |
| w4a4_smoothquant | 0.3193 |
| w4a4_hadamard | 0.2195 |
| w4a4_hadamard_gptq | 0.1714 |
| w4a4_mxfp4 | 0.1624 |
| w4a4_hadamard_mxfp4 | 0.1734 |
| w4a4_nvfp4 | 0.1195 |
| w4a4_hadamard_nvfp4 | 0.1433 |
