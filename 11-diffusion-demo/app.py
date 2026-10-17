"""My version of the Gemma Diffusion website-builder demo (Task 11).

A prompt box, a code view that updates on every denoising step (you can watch each 256-token canvas
fill in), a live preview of the page in a sandboxed iframe, and a table of how many denoising steps each
canvas needed. Run it from the notebook (it needs a big GPU) or with a fake stream function for testing.
"""
import html

import gradio as gr

import core

EXAMPLES = [
    "A landing page for a small coffee shop called Bean There, with a menu and opening hours.",
    "A to-do list app where I can add, tick off and delete tasks.",
    "A pomodoro timer with start, pause and reset buttons.",
]
PREVIEW_EVERY = 4  # refresh the preview every N denoising steps (every step makes the iframe flicker)


def iframe(code):
    """A sandboxed iframe: the generated page can run its own JavaScript but cannot reach this app."""
    if not code.strip():
        return "<div style='padding:2em;color:#888'>The preview appears here once the page has started.</div>"
    return ("<iframe sandbox='allow-scripts' style='width:100%;height:520px;border:1px solid #ccc;"
            f"border-radius:6px;background:white' srcdoc=\"{html.escape(code, quote=True)}\"></iframe>")


def table(rows):
    return [[r["canvas"], r["steps"], r["seconds"]] for r in rows]


def make_handler(stream_fn):
    """stream_fn(prompt, max_new_tokens, max_denoising_steps) yields core-style frames."""

    def handler(prompt, max_new_tokens, max_steps):
        prompt = (prompt or "").strip()
        if not prompt:
            yield "", iframe(""), [], "Type a request first."
            return
        text, rows, last_preview, n_draft = "", [], -99, 0
        status = "Starting..."
        yield text, iframe(""), rows, status
        for f in stream_fn(prompt, int(max_new_tokens), int(max_steps)):
            k = f["kind"]
            if k == "error":
                yield text, iframe(core.extract_html(text)), table(rows), "Error: " + f["message"]
                return
            if k == "draft":
                n_draft += 1
                text = f["text"]
                status = f"Canvas {f['canvas'] + 1}, denoising step {f['step']}"
                preview = None
                if n_draft - last_preview >= PREVIEW_EVERY:
                    preview, last_preview = iframe(core.extract_html(text)), n_draft
                yield text, (preview if preview is not None else gr.update()), table(rows), status
            elif k == "commit":
                text = f["text"]
                rows.append(f["info"])
                status = f"Canvas {f['canvas'] + 1} finished after {f['info']['steps']} steps"
                yield text, iframe(core.extract_html(text)), table(rows), status
            elif k == "done":
                status = (f"Done: {f['new_tokens']} tokens in {f['seconds']} s = {f['tokens_per_s']} tokens/s "
                          f"({len(f['canvases'])} canvases)")
                yield text, iframe(core.extract_html(text)), table(rows), status

    return handler


def build_ui(stream_fn):
    with gr.Blocks(title="Diffusion website builder") as demo:
        gr.Markdown("## Diffusion website builder\nDiffusionGemma writes the page 256 tokens at a time. "
                    "Watch each block of code get filled in and sharpened over several denoising steps.")
        prompt = gr.Textbox(label="What page should it build?", lines=2, value=EXAMPLES[0])
        with gr.Row():
            max_new = gr.Slider(256, 2048, value=1024, step=256, label="Max new tokens (canvases of 256)")
            max_steps = gr.Slider(4, 48, value=48, step=4, label="Max denoising steps per canvas")
        go = gr.Button("Build it", variant="primary")
        gr.Examples(EXAMPLES, inputs=prompt)
        status = gr.Markdown("")
        with gr.Row():
            code = gr.Textbox(label="Code (live, every denoising step)", lines=24, max_lines=24)
            with gr.Column():
                preview = gr.HTML(iframe(""), label="Preview")
                steps = gr.Dataframe(headers=["canvas", "denoising steps used", "seconds"],
                                     label="Steps per canvas", interactive=False)
        go.click(make_handler(stream_fn), [prompt, max_new, max_steps], [code, preview, steps, status])
    return demo


STATE = {}  # the loaded model lives here (not inside the app) so stop() can really free the GPU


def launch(model, processor, **kwargs):
    STATE.update(model=model, processor=processor)

    def stream_fn(prompt, max_new_tokens, max_steps):
        return core.stream_generate(STATE["model"], STATE["processor"], prompt, max_new_tokens=max_new_tokens,
                                    max_denoising_steps=max_steps, as_page=True)
    STATE["demo"] = demo = build_ui(stream_fn)
    demo.queue()
    return demo.launch(**kwargs)


def stop():
    """Close the app and drop its reference to the model."""
    if "demo" in STATE:
        STATE["demo"].close()
    STATE.clear()
