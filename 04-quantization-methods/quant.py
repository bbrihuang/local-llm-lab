"""The quantization methods compared in Task 4, written as "fake quantization" in plain PyTorch.

Every method rounds the numbers exactly the way the real format would (4-bit integers with a scale,
FP4 values with a shared exponent, and so on) and then turns them back into ordinary floats. The model
runs at the same speed as before, but its outputs carry the real rounding error, which is what we want
to measure. None of this makes anything faster or smaller; real kernels do that.

Notation: a linear layer computes y = x W^T, with W of shape (out, in) and x of shape (tokens, in).
"Per channel" means one scale per output row of W; "per token" means one scale per row of x;
"group 128" means one scale per 128 consecutive input numbers.
"""
import math

import torch
from torch import nn

# ----------------------------------------------------------------------------- number formats
# Each format has params(block, ctx) -> scales for each row of the block, and quant(x, params) -> rounded x.


class IntAsym:
    """Unsigned integers with a scale and a zero point (min/max range). What GPTQ and AWQ use for weights."""

    def __init__(self, bits):
        self.maxq = 2 ** bits - 1

    def context(self, x):
        return None

    def params(self, w, ctx=None):
        lo = w.amin(1, keepdim=True).clamp(max=0)
        hi = w.amax(1, keepdim=True).clamp(min=0)
        scale = ((hi - lo) / self.maxq).clamp(min=1e-10)
        return scale, torch.round(-lo / scale)

    def quant(self, w, p):
        scale, zero = p
        return scale * (torch.clamp(torch.round(w / scale) + zero, 0, self.maxq) - zero)


class IntSym:
    """Signed integers, scale = largest absolute value / 127 (for 8 bits). Used when activations are quantized too."""

    def __init__(self, bits):
        self.qmax = 2 ** (bits - 1) - 1

    def context(self, x):
        return None

    def params(self, w, ctx=None):
        return (w.abs().amax(1, keepdim=True) / self.qmax).clamp(min=1e-10)

    def quant(self, w, s):
        return torch.clamp(torch.round(w / s), -self.qmax, self.qmax) * s


FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # every positive number FP4 (E2M1) can hold


def round_fp4(x):
    """Round to the nearest FP4 (E2M1) value, keeping the sign. Anything above 6 becomes 6."""
    grid = FP4_VALUES.to(x.device, x.dtype)
    idx = torch.bucketize(x.abs().clamp(max=6.0), (grid[1:] + grid[:-1]) / 2)
    return grid[idx] * torch.sign(x)


class MXFP4:
    """OCP microscaling FP4: blocks of 32 numbers share one power-of-two scale (an 8-bit exponent).

    Scale = 2^(floor(log2(largest |x| in the block)) - 2), as in the OCP MX spec. Because the scale can only be a
    power of two, the largest number in a block often lands above 6 and gets clipped.
    """
    block = 32

    def context(self, x):
        return None

    def params(self, w, ctx=None):
        amax = w.abs().amax(1, keepdim=True).clamp(min=2.0 ** -120)
        return torch.exp2(torch.floor(torch.log2(amax)) - 2)

    def quant(self, w, s):
        return round_fp4(w / s) * s


class NVFP4:
    """NVIDIA FP4: blocks of 16 numbers share an FP8 (E4M3) scale, plus one FP32 scale for the whole tensor."""
    block = 16

    def context(self, x):
        return (x.abs().max() / (6 * 448)).clamp(min=1e-12)

    def params(self, w, g):
        s = (w.abs().amax(1, keepdim=True) / 6 / g).clamp(max=448)
        return (s.to(torch.float8_e4m3fn).float() * g).clamp(min=1e-12)

    def quant(self, w, s):
        return round_fp4(w / s) * s


def fake_quant(x, fmt, group=None):
    """Quantize the last dimension of x. group=None: one scale per row; 'tensor': one scale for everything;
    a number: one scale per that many consecutive values."""
    shape = x.shape
    ctx = fmt.context(x)
    if group == "tensor":
        xg = x.reshape(1, -1)
    elif group is None:
        xg = x.reshape(-1, shape[-1])
    else:
        xg = x.reshape(-1, group)
    return fmt.quant(xg, fmt.params(xg, ctx)).reshape(shape)


def ternary(W):
    """BitNet b1.58: every weight becomes -1, 0 or +1 times the mean absolute weight of the whole matrix."""
    g = W.abs().mean().clamp(min=1e-10)
    return torch.clamp(torch.round(W / g), -1, 1) * g


# ----------------------------------------------------------------------------- rotations (QuaRot)
_HADAMARD = {}


def hadamard(n, device, seed=0):
    """An orthonormal random Hadamard matrix: random signs times a Sylvester Hadamard matrix, scaled by 1/sqrt(n).

    Multiplying by it spreads one big number over all positions, which removes outliers. When n is not a power
    of two (3072 for Qwen3-0.6B's down_proj) it is block-diagonal with the largest power-of-two block that fits.
    """
    key = (n, str(device), seed)
    if key not in _HADAMARD:
        b = n & (-n)
        H = torch.ones(1, 1)
        while H.shape[0] < b:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        H = torch.block_diag(*[H] * (n // b)) / math.sqrt(b)
        signs = torch.randint(0, 2, (n,), generator=torch.Generator().manual_seed(seed)) * 2 - 1
        _HADAMARD[key] = (signs[:, None] * H).to(device)
    return _HADAMARD[key]


# ----------------------------------------------------------------------------- GPTQ and AWQ
def out_error(dW, H):
    """Squared output error summed over the calibration tokens: sum ||dW x||^2 = trace(dW H dW^T)."""
    return ((dW @ H) * dW).sum()


def gptq(W, H, fmt, group=None, damp=0.01, block=128):
    """GPTQ (Frantar et al., 2022): quantize one input column at a time and spread its rounding error over the
    columns not yet quantized, using the inverse of H = sum of x x^T over calibration tokens.
    Follows the reference implementation (Cholesky form of the inverse, lazy block updates)."""
    W = W.clone().float()
    H = H.clone().float()
    n = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(n, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    ctx = fmt.context(W)
    g = group or n
    Q = torch.zeros_like(W)
    p = None
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if (i1 + i) % g == 0:  # new group: pick its scales from the columns as they are now
                p = fmt.params(torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :g], ctx)
            w = W1[:, i]
            q = fmt.quant(w[:, None], p)[:, 0]
            Q[:, i1 + i] = q
            e = (w - q) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None, :]
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


def awq_scales(W, H, xmean, fmt, group, grid=20):
    """AWQ (Lin et al., 2023): scale input channel j of W up by s_j = mean|x_j|^a (and x down by the same),
    so the weights that meet big activations get relatively finer rounding. Search a in 0, 0.05, ..., 0.95
    for the lowest output error. (The real AWQ also searches a clipping range; this does not.)"""
    best, best_s = None, None
    for k in range(grid):
        s = xmean.clamp(min=1e-4) ** (k / grid)
        s = s / (s.max() * s.min()).sqrt()
        err = out_error(W - fake_quant(W * s, fmt, group) / s, H)
        if best is None or err < best:
            best, best_s = err, s
    return best_s


def smooth_scales(xmax, W, alpha=0.5):
    """SmoothQuant (Xiao et al., 2022): s_j = max|x_j|^alpha / max|W_j|^(1-alpha). Dividing x by s and multiplying
    W by s moves the outliers from the activations into the weights, where they are easier to handle."""
    return (xmax.clamp(min=1e-5) ** alpha / W.abs().amax(0).clamp(min=1e-5) ** (1 - alpha)).clamp(min=1e-5)


# ----------------------------------------------------------------------------- the methods
# name: (family, bits per weight including scales, what it does)
METHODS = {
    "fp": ("baseline", 16, "no quantization"),
    "w4_rtn_tensor": ("W4A16", 4.0, "round to nearest, one scale for the whole matrix"),
    "w4_rtn_channel": ("W4A16", 4.0, "round to nearest, one scale per output row"),
    "w4_rtn_g128": ("W4A16", 4.25, "round to nearest, one scale per 128 weights"),
    "w4_gptq_g128": ("W4A16", 4.25, "GPTQ error correction, group 128"),
    "w4_awq_g128": ("W4A16", 4.25, "AWQ activation-aware scaling, group 128"),
    "w4_hadamard_g128": ("W4A16", 4.25, "Hadamard rotation, then round to nearest, group 128"),
    "w4_mxfp4": ("W4A16", 4.25, "MXFP4 weights (block 32, power-of-two scale)"),
    "w4_nvfp4": ("W4A16", 4.5, "NVFP4 weights (block 16, FP8 scale)"),
    "w1.58_ternary": ("W1.58", 1.58, "BitNet b1.58 ternary weights, applied after training"),
    "w8a8_tensor": ("W8A8", 8, "int8 weights per row, int8 activations with one scale per batch"),
    "w8a8_token": ("W8A8", 8, "int8 weights per row, int8 activations per token"),
    "w8a8_llm_int8": ("W8A8", 8, "LLM.int8(): per-token int8, but outlier features (|x| > 6) kept in 16 bits"),
    "w8a8_smoothquant": ("W8A8", 8, "SmoothQuant (alpha 0.5), then per-token int8"),
    "w4a4_token": ("W4A4", 4, "int4 weights per row, int4 activations per token"),
    "w4a4_smoothquant": ("W4A4", 4, "SmoothQuant (alpha 0.5), then int4"),
    "w4a4_hadamard": ("W4A4", 4, "Hadamard rotation, then int4 (QuaRot without GPTQ)"),
    "w4a4_hadamard_gptq": ("W4A4", 4, "Hadamard rotation, GPTQ int4 weights (the QuaRot recipe)"),
    "w4a4_mxfp4": ("W4A4", 4.25, "MXFP4 weights and activations"),
    "w4a4_hadamard_mxfp4": ("W4A4", 4.25, "Hadamard rotation, then MXFP4"),
    "w4a4_nvfp4": ("W4A4", 4.5, "NVFP4 weights and activations"),
    "w4a4_hadamard_nvfp4": ("W4A4", 4.5, "Hadamard rotation, then NVFP4"),
}


class QLinear(nn.Module):
    """A stand-in for nn.Linear that holds already-rounded weights and rounds its input on the way in."""

    def __init__(self, Wq, bias=None, act=None, rot=None, smooth=None, W_outlier=None):
        super().__init__()
        self.Wq, self.bias = Wq, bias            # Wq: (out, in), float32
        self.act = act                           # (format, group) for the activations, or None
        self.rot, self.smooth = rot, smooth      # input rotation / SmoothQuant divisor, or None
        self.W_outlier = W_outlier               # LLM.int8: the 16-bit weights used for outlier features

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        if self.smooth is not None:
            x = x / self.smooth
        if self.rot is not None:
            x = x @ self.rot
        if self.W_outlier is not None:          # LLM.int8(): split by feature
            out = x.reshape(-1, x.shape[-1]).abs().amax(0) > 6.0
            x_out = x * out
            y = fake_quant(x - x_out, *self.act) @ self.Wq.T + x_out @ self.W_outlier.float().T
        else:
            if self.act is not None:
                x = fake_quant(x, *self.act)
            y = x @ self.Wq.T
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(dtype)


def make_qlinear(method, W, bias=None, st=None):
    """Build the quantized layer for one method. W is the original weight; st holds calibration statistics
    of this layer's input: H (sum of x x^T), xmax (max |x| per feature) and xmean (mean |x| per feature)."""
    W = W.detach().float()
    n = W.shape[1]
    int4a, int4s, int8s = IntAsym(4), IntSym(4), IntSym(8)
    g128 = 128 if n % 128 == 0 else None
    if method == "fp":
        return QLinear(W, bias)
    if method == "w4_rtn_tensor":
        return QLinear(fake_quant(W, int4a, "tensor"), bias)
    if method == "w4_rtn_channel":
        return QLinear(fake_quant(W, int4a), bias)
    if method == "w4_rtn_g128":
        return QLinear(fake_quant(W, int4a, g128), bias)
    if method == "w4_gptq_g128":
        return QLinear(gptq(W, st["H"], int4a, g128), bias)
    if method == "w4_awq_g128":
        s = awq_scales(W, st["H"], st["xmean"], int4a, g128)
        return QLinear(fake_quant(W * s, int4a, g128) / s, bias)
    if method == "w4_hadamard_g128":           # rotate, round, rotate back (same output as rotating x at run time)
        R = hadamard(n, W.device)
        return QLinear(fake_quant(W @ R, int4a, g128) @ R.T, bias)
    if method == "w4_mxfp4":
        return QLinear(fake_quant(W, MXFP4(), 32), bias)
    if method == "w4_nvfp4":
        return QLinear(fake_quant(W, NVFP4(), 16), bias)
    if method == "w1.58_ternary":
        return QLinear(ternary(W), bias)
    if method == "w8a8_tensor":
        return QLinear(fake_quant(W, int8s), bias, act=(int8s, "tensor"))
    if method == "w8a8_token":
        return QLinear(fake_quant(W, int8s), bias, act=(int8s, None))
    if method == "w8a8_llm_int8":
        return QLinear(fake_quant(W, int8s), bias, act=(int8s, None), W_outlier=W.half())
    if method in ("w8a8_smoothquant", "w4a4_smoothquant"):
        q = int8s if method.startswith("w8") else int4s
        s = smooth_scales(st["xmax"], W)
        return QLinear(fake_quant(W * s, q), bias, act=(q, None), smooth=s)
    if method == "w4a4_token":
        return QLinear(fake_quant(W, int4s), bias, act=(int4s, None))
    if method == "w4a4_hadamard":
        R = hadamard(n, W.device)
        return QLinear(fake_quant(W @ R, int4s), bias, act=(int4s, None), rot=R)
    if method == "w4a4_hadamard_gptq":
        R = hadamard(n, W.device)
        return QLinear(gptq(W @ R, R.T @ st["H"] @ R, int4s), bias, act=(int4s, None), rot=R)
    if method in ("w4a4_mxfp4", "w4a4_nvfp4", "w4a4_hadamard_mxfp4", "w4a4_hadamard_nvfp4"):
        fmt, blk = (MXFP4(), 32) if "mxfp4" in method else (NVFP4(), 16)
        R = hadamard(n, W.device) if "hadamard" in method else None
        Wr = W @ R if R is not None else W
        return QLinear(fake_quant(Wr, fmt, blk), bias, act=(fmt, blk), rot=R)
    raise ValueError(f"unknown method {method}")


# ----------------------------------------------------------------------------- calibration statistics
class InputStats:
    """Running statistics of one linear layer's input."""

    def __init__(self, n, device, want_H=True):
        self.H = torch.zeros(n, n, device=device) if want_H else None
        self.xmax = torch.zeros(n, device=device)
        self.xsum = torch.zeros(n, device=device)
        self.count = 0

    def add(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.H is not None:
            self.H += x.T @ x
        self.xmax = torch.maximum(self.xmax, x.abs().amax(0))
        self.xsum += x.abs().sum(0)
        self.count += x.shape[0]

    def as_dict(self):
        return {"H": self.H, "xmax": self.xmax, "xmean": self.xsum / max(self.count, 1)}


def input_key(name):
    """q/k/v_proj read the same input, and so do gate/up_proj, so they share one set of statistics."""
    return name.replace("k_proj", "q_proj").replace("v_proj", "q_proj").replace("up_proj", "gate_proj")


def linear_layers(model):
    """All the linear layers inside the transformer blocks (the embedding and output head stay in 16 bits)."""
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and ".layers." in n}


@torch.no_grad()
def collect_stats(model, batches, names, want_H=True, capture=None):
    """Run the model over the calibration batches and record input statistics for the named linear layers.
    capture: optional set of names whose raw inputs are also kept (for the per-layer test)."""
    layers = linear_layers(model)
    stats, kept, hooks = {}, {}, []
    for name in names:
        key = input_key(name)
        if key not in stats:
            stats[key] = InputStats(layers[name].in_features, next(model.parameters()).device, want_H)

            def hook(mod, inp, out, key=key):
                stats[key].add(inp[0])
            hooks.append(layers[key].register_forward_hook(hook))
    if capture:
        for name in capture:
            def keep(mod, inp, out, name=name):
                kept.setdefault(name, []).append(inp[0].reshape(-1, inp[0].shape[-1]).float())
            hooks.append(layers[name].register_forward_hook(keep))
    try:
        for ids in batches:
            model(ids[None].to(next(model.parameters()).device))
    finally:
        for h in hooks:
            h.remove()
    return {k: s.as_dict() for k, s in stats.items()}, {k: torch.cat(v) for k, v in kept.items()}


def set_module(model, name, new):
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent), child, new)


# ----------------------------------------------------------------------------- shared by the two scripts
FAMILY_COLORS = {"W4A16": "#2a6ebb", "W1.58": "#d9822b", "W8A8": "#2a9d8f", "W4A4": "#8e5ea2", "baseline": "#888888"}


def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        print("WARNING: PyTorch can't see the GPU, so this will run on the CPU and be slow.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {name} ...")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
    return model, tok


def token_chunks(tok, path, length, count, max_chars=None):
    """Tokenize a text file and cut it into `count` pieces of `length` tokens."""
    import os
    if not os.path.exists(path):
        raise SystemExit(f"Can't find {path}. Check the WikiText-2 folder (it should hold wiki.train.raw and wiki.test.raw).")
    with open(path, encoding="utf-8") as f:
        text = f.read(max_chars) if max_chars else f.read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    if len(ids) < length * count:
        raise SystemExit(f"{path} has only {len(ids):,} tokens; need {length * count:,}.")
    return [ids[i * length:(i + 1) * length] for i in range(count)]
