"""Checks for quant.py on a made-up layer (no model download, runs on the CPU in a few seconds).

Run from the 04-quantization-methods folder:  ..\\.venv\\Scripts\\python tests\\test_quant.py
The made-up inputs have a few huge features, like real LLM activations, so the outlier methods should win.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import quant as Q  # noqa: E402

torch.manual_seed(0)
n, m, T = 512, 256, 8192
A = torch.randn(n, n) / n ** 0.5
X = torch.randn(T, n) @ (torch.eye(n) + 2 * A) + 2 * torch.randn(T, 1) * torch.randn(n)  # correlated features
X[:, [7, 300]] *= 40                                                                        # two outlier features
W = torch.randn(m, n) * 0.02
st = Q.InputStats(n, "cpu")
st.add(X[:4096])
st = st.as_dict()
Xe, Y = X[4096:], X[4096:] @ W.T

err = {}
for method in Q.METHODS:
    with torch.no_grad():
        err[method] = ((Q.make_qlinear(method, W, None, st)(Xe) - Y).norm() / Y.norm()).item()
    print(f"{method:22} {err[method]:.4f}")

assert err["fp"] < 1e-6
assert err["w4_rtn_channel"] < err["w4_rtn_tensor"]
assert err["w4_gptq_g128"] < err["w4_rtn_g128"]          # GPTQ uses the correlations between features
assert err["w8a8_llm_int8"] < err["w8a8_token"]           # keeping the outliers in 16 bits helps
assert err["w8a8_smoothquant"] < err["w8a8_token"]
assert err["w4a4_hadamard"] < err["w4a4_token"]           # rotation spreads the outliers out
assert err["w4a4_hadamard_gptq"] < err["w4a4_hadamard"]
assert set(Q.round_fp4(torch.randn(1000) * 4).abs().unique().tolist()) <= set(Q.FP4_VALUES.tolist())
R = Q.hadamard(768, "cpu")
assert torch.allclose(R @ R.T, torch.eye(768), atol=1e-5)  # a rotation: changes nothing on its own
print("all checks passed")
