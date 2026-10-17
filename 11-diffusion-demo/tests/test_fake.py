"""Tests with a fake model that follows the same streamer protocol as DiffusionGemma's generate().

Run:  python tests/test_fake.py        (from the 11-diffusion-demo folder; needs torch, and gradio for the app test)
This only proves OUR code handles the protocol correctly. It says nothing about the real model.
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import core  # noqa: E402

TEXT = "```html\n<h1>Hi</h1>\n<p>diffusion</p>\n```"
CANVAS = 16
EOS = 1


class FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i > 3)


class FakeProc:
    tokenizer = FakeTok()

    def apply_chat_template(self, messages, enable_thinking=False, **kw):
        class In(dict):
            def to(self, d):
                return self
        return In(input_ids=torch.tensor([[2, 5, 6, 7]]), attention_mask=torch.ones(1, 4, dtype=torch.long))


class FakeModel:
    device = "cpu"
    generation_config = types.SimpleNamespace(eos_token_id=[EOS], pad_token_id=0)

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=32, max_denoising_steps=4, streamer=None, **kw):
        ids = [ord(c) for c in TEXT] + [EOS]
        n_canvas = -(-min(len(ids), max_new_tokens) // CANVAS)
        ids += [0] * (n_canvas * CANVAS - len(ids))
        if streamer is None:
            streamer = types.SimpleNamespace(put=lambda v: None, put_draft=lambda value: None, end=lambda: None)
        streamer.put(input_ids)
        out = input_ids
        for c in range(n_canvas):
            target = torch.tensor(ids[c * CANVAS:(c + 1) * CANVAS])
            for s in range(1, max_denoising_steps + 1):
                draft = target.clone()
                draft[torch.arange(CANVAS) >= CANVAS * s // max_denoising_steps] = 9  # unfinished tail
                streamer.put_draft(value=draft[None])
            streamer.put(target[None])
            out = torch.cat([out, target[None]], 1)
        streamer.end()
        return types.SimpleNamespace(sequences=out, tokens_per_forward=3.0)


def test_core():
    frames = list(core.stream_generate(FakeModel(), FakeProc(), "make a page", 64, 4))
    kinds = [f["kind"] for f in frames]
    n_canvas = -(-(len(TEXT) + 1) // CANVAS)
    assert kinds.count("draft") == 4 * n_canvas, kinds
    assert kinds.count("commit") == n_canvas
    done = frames[-1]
    assert done["kind"] == "done" and done["new_tokens"] == len(TEXT) + 1, done
    assert [c["steps"] for c in done["canvases"]] == [4] * n_canvas
    assert frames[-2]["kind"] == "commit" and frames[-2]["text"] == TEXT
    assert core.extract_html(frames[-2]["text"]) == "<h1>Hi</h1>\n<p>diffusion</p>\n"
    assert core.extract_html("```html\n<b>half").startswith("<b>")   # unfinished fence still previews
    assert core.extract_html("no code here") == ""
    print("core ok:", done["new_tokens"], "tokens,", n_canvas, "canvases")


def test_error():
    class Bad(FakeModel):
        def generate(self, **kw):
            raise RuntimeError("out of memory")
    frames = list(core.stream_generate(Bad(), FakeProc(), "x", 64, 4))
    assert frames[-1]["kind"] == "error" and "out of memory" in frames[-1]["message"], frames
    print("error path ok")


def test_benchmark(tmp):
    import benchmark
    for kind in ("diffusion", "regular"):
        rows = benchmark.run(kind, FakeModel(), FakeProc(), 64, 2, tmp)
        assert rows[0]["new_tokens_median"] == len(TEXT) + 1
    assert rows and "steps_per_canvas" not in rows[0]
    benchmark.combine(tmp)
    assert os.path.exists(os.path.join(tmp, "bench_speed.png"))
    print("benchmark ok")


def test_app():
    import app
    h = app.make_handler(lambda p, n, s: core.stream_generate(FakeModel(), FakeProc(), p, n, s))
    outs = list(h("a page", 64, 4))
    code, preview, rows, status = outs[-1]
    assert code == TEXT and "srcdoc" in preview and "&lt;h1&gt;Hi" in preview
    assert len(rows) == -(-(len(TEXT) + 1) // CANVAS) and status.startswith("Done"), (rows, status)
    assert list(h("  ", 64, 4))[0][3] == "Type a request first."
    app.build_ui(lambda p, n, s: iter(()))
    print("app ok:", status)


if __name__ == "__main__":
    test_core()
    test_error()
    import tempfile
    test_benchmark(tempfile.mkdtemp())
    try:
        test_app()
    except ImportError:
        print("gradio not installed - app test skipped")
