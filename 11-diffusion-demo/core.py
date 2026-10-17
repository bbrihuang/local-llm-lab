"""Shared code for the DiffusionGemma demo (Task 11).

Everything that touches the model lives here so the app, the benchmark and the tests share it.
Written against the transformers source for DiffusionGemma:
  * model.generate(...) works canvas by canvas (256 tokens each). For every denoising step it calls
    streamer.put_draft(value=<the current best guess for the canvas>); when a canvas is finished it calls
    streamer.put(<the 256 confirmed tokens>). The very first put() call is the prompt.
  * the returned object has .sequences (prompt + new tokens) and .tokens_per_forward.
"""
import gc
import queue
import re
import threading
import time

import torch

try:  # BaseStreamer only matters for isinstance checks inside transformers; generate() just calls the methods.
    from transformers.generation.streamers import BaseStreamer
except Exception:  # lets the tests run without transformers installed
    BaseStreamer = object

DIFFUSION_ID = "google/diffusiongemma-26B-A4B-it"
REGULAR_ID = "google/gemma-4-26B-A4B-it"

PAGE_PROMPT = (
    "Write a complete, self-contained HTML page (one file, inline CSS and JavaScript, no external "
    "files or libraries) for this request:\n\n{request}\n\n"
    "Reply with only one ```html code block."
)


# --------------------------------------------------------------------------- loading
def load_model(kind, dtype="auto"):
    """Load 'diffusion' or 'regular' in bf16 on the GPU. Returns (model, processor)."""
    from transformers import AutoProcessor
    if kind == "diffusion":
        from transformers import DiffusionGemmaForBlockDiffusion as Cls
        model_id = DIFFUSION_ID
    else:
        from transformers import AutoModelForMultimodalLM as Cls
        model_id = REGULAR_ID
    processor = AutoProcessor.from_pretrained(model_id)
    model = Cls.from_pretrained(model_id, dtype=dtype, device_map="cuda")
    model.eval()
    return model, processor


def free_gpu():
    """Give back GPU memory. The caller must first delete its own references to the model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- inputs
def build_inputs(processor, model, prompt, as_page=False, thinking=False):
    """Chat-template the prompt and move it to the model's device. Returns (inputs, prompt_length)."""
    text = PAGE_PROMPT.format(request=prompt) if as_page else prompt
    messages = [{"role": "user", "content": text}]
    kwargs = dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    try:
        inputs = processor.apply_chat_template(messages, enable_thinking=thinking, **kwargs)
    except TypeError:
        inputs = processor.apply_chat_template(messages, **kwargs)
    except Exception:  # some multimodal templates want the content as a list of parts
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        inputs = processor.apply_chat_template(messages, enable_thinking=thinking, **kwargs)
    inputs = inputs.to(model.device)
    return inputs, inputs["input_ids"].shape[1]


def _eos_ids(model):
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        return set()
    return set(eos) if isinstance(eos, (list, tuple)) else {eos}


def count_new_tokens(sequences, prompt_len, model):
    """Tokens written after the prompt, up to and including the first end-of-sequence token."""
    new = sequences[0, prompt_len:].tolist()
    eos = _eos_ids(model)
    for i, t in enumerate(new):
        if t in eos:
            return i + 1
    pad = getattr(model.generation_config, "pad_token_id", None)
    while new and pad is not None and new[-1] == pad:
        new.pop()
    return len(new)


# --------------------------------------------------------------------------- streaming the denoising
class FrameStreamer(BaseStreamer):
    """Turns generate()'s streamer calls into frames (dicts) on a queue.

    Frame kinds:  'draft'  one denoising step of the current canvas
                  'commit' a canvas is finished (its tokens are now final)
                  'end'    generation finished
    """

    def __init__(self, tokenizer, out_queue, keep_text=True):
        self.tok = tokenizer
        self.q = out_queue
        self.keep_text = keep_text
        self.confirmed = []          # token ids of finished canvases
        self.canvas = 0              # index of the canvas being denoised
        self.step = 0                # denoising steps so far in this canvas
        self.canvas_t0 = None
        self.canvases = []           # one dict per finished canvas
        self._prompt_seen = False
        self._t_start = time.perf_counter()

    def _text(self, ids):
        if not self.keep_text:
            return ""
        # even with thinking off, the template leaves a bare "thought" line at the start of the answer
        return re.sub(r"^\s*thought\s*\n", "", self.tok.decode(ids, skip_special_tokens=True))

    def put_draft(self, value=None, **kwargs):
        if self.canvas_t0 is None:
            self.canvas_t0 = time.perf_counter()
        self.step += 1
        ids = value[0].tolist() if value.dim() > 1 else value.tolist()
        self.q.put({"kind": "draft", "canvas": self.canvas, "step": self.step,
                    "text": self._text(self.confirmed + ids)})

    def put(self, value):
        if not self._prompt_seen:      # first call is the prompt: nothing to show
            self._prompt_seen = True
            return
        ids = value[0].tolist() if value.dim() > 1 else value.tolist()
        self.confirmed += ids
        now = time.perf_counter()
        info = {"canvas": self.canvas + 1, "steps": self.step,
                "seconds": round(now - (self.canvas_t0 or now), 3)}
        self.canvases.append(info)
        self.q.put({"kind": "commit", "canvas": self.canvas, "step": self.step,
                    "text": self._text(self.confirmed), "info": info})
        self.canvas += 1
        self.step = 0
        self.canvas_t0 = None

    def end(self):
        self.q.put({"kind": "end"})


def stream_generate(model, processor, prompt, max_new_tokens=1024, max_denoising_steps=48,
                    as_page=True, thinking=False):
    """Run DiffusionGemma in a worker thread and yield frames as they arrive.

    The last frame has kind 'done' and carries totals: new_tokens, seconds, tokens_per_s, canvases.
    """
    q = queue.Queue()
    inputs, prompt_len = build_inputs(processor, model, prompt, as_page, thinking)
    streamer = FrameStreamer(processor.tokenizer, q)
    result = {}

    def work():
        try:
            with torch.inference_mode():
                t0 = time.perf_counter()
                out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     max_denoising_steps=max_denoising_steps, streamer=streamer)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                result["seconds"] = time.perf_counter() - t0
                result["out"] = out
        except Exception as e:  # shown in the UI instead of silently dying
            q.put({"kind": "error", "message": f"{type(e).__name__}: {e}"})
            result["error"] = True
            q.put({"kind": "end"})

    th = threading.Thread(target=work, daemon=True)
    th.start()
    while True:
        frame = q.get()
        if frame["kind"] == "end":
            break
        yield frame
    th.join()
    if "out" in result:
        seq = getattr(result["out"], "sequences", result["out"])
        n = count_new_tokens(seq, prompt_len, model)
        sec = result["seconds"]
        yield {"kind": "done", "new_tokens": n, "seconds": round(sec, 2),
               "tokens_per_s": round(n / sec, 1) if sec else None,
               "canvases": streamer.canvases,
               "tokens_per_forward": getattr(result["out"], "tokens_per_forward", None)}


# --------------------------------------------------------------------------- turning text into a preview
def extract_html(text):
    """Pull the HTML out of a ```html fence (also an unfinished one) or return the text if it looks like HTML."""
    m = re.search(r"```(?:html)?\s*\n(.*?)(?:```|\Z)", text, re.S | re.I)
    code = m.group(1) if m else text
    return code if "<" in code else ""
