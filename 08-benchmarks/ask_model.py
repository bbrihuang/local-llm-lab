"""Ask Qwen3.5-4B (Task 3's Q4_K_M file) the 30 sample questions, with thinking off and on.

Run from the 08-benchmarks folder, after sample_questions.py:
    ..\\.venv\\Scripts\\python ask_model.py

Starts llama-server, asks every question in questions/questions.json once with thinking off
("instant") and once with thinking on ("thinking"), checks the answers, and saves
results/model_answers.csv. Answers to GPQA and Humanity's Last Exam questions are saved
without the question text (their authors ask that questions not be posted).
"""
import argparse
import csv
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--server", default=r"C:\Users\brian\tools\llama.cpp\llama-server.exe")
ap.add_argument("--model", default=r"C:\Users\brian\models\Qwen3.5-4B-Q4_K_M.gguf")
ap.add_argument("--modes", default="instant,thinking")
ap.add_argument("--only", default="", help="only these benchmarks, comma-separated (e.g. RULER,IFEval)")
ap.add_argument("--port", type=int, default=8080)
args = ap.parse_args()
URL = f"http://127.0.0.1:{args.port}"
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# Qwen's recommended sampling for each mode (Qwen3.5-4B model card), same as run_harness.py.
MODES = {
    "instant": dict(thinking=False, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=4096),
    "thinking": dict(thinking=True, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=20000),
}


def health():
    try:
        with DIRECT.open(URL + "/health", timeout=2) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


def ask(prompt, mode):
    m = dict(MODES[mode])
    thinking = m.pop("thinking")
    body = dict(model="qwen3.5-4b", messages=[{"role": "user", "content": prompt}], seed=1234,
                chat_template_kwargs={"enable_thinking": thinking}, **m)
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with DIRECT.open(req, timeout=3600) as r:
        out = json.loads(r.read())
    msg = out["choices"][0]["message"]
    return (msg.get("content") or "", msg.get("reasoning_content") or "", out["choices"][0].get("finish_reason"),
            out.get("usage", {}), time.perf_counter() - t0)


def normalize(s):
    return re.sub(r"[^a-z0-9.]+", " ", str(s).lower()).strip()


def extract(q, text):
    if q["kind"] == "mc":
        found = re.findall(r"answer is \(?([A-J])\)?", text)
        return found[-1] if found else ""
    found = re.findall(r"final answer:\s*(.+)", text, flags=re.I)
    return (found[-1] if found else text).strip().strip("*").strip()


def check(q, given, full_text):
    """1 = right, 0 = wrong, "" = needs checking by hand."""
    if q["kind"] == "mc":
        return int(given == q["answer"])
    if q["kind"] == "ifeval":
        from lm_eval.tasks.ifeval.utils import process_results
        doc = {k: q[k] for k in ("instruction_id_list", "kwargs")}
        doc.update(key=q["id"], prompt=q["prompt"])
        return int(process_results(doc, [full_text])["prompt_level_strict_acc"])
    if q["kind"] == "ruler" and q["id"] == "variable_tracking":
        return int(set(q["answer"].split()) == set(re.findall(r"\b[A-Z]{5}\b", given)))
    if q.get("answer_type") == "multipleChoice":
        return int(normalize(given)[:1] == normalize(q["answer"])[:1])
    # Short answers: exact (after tidying) counts as right; anything else gets a look by hand.
    a, g = normalize(q["answer"]), normalize(given)
    if a == g or (a and re.search(rf"\b{re.escape(a)}\b", g)):
        return 1
    return 0 if q["kind"] == "ruler" else ""


questions = json.load(open("questions/questions.json", encoding="utf-8"))
if args.only:
    questions = [q for q in questions if q["benchmark"] in args.only.split(",")]
modes = args.modes.split(",")

if health() is not None:
    raise SystemExit(f"Something is already using port {args.port}. Close other llama-server windows, "
                     "or run: taskkill /F /IM llama-server.exe")
for path in (args.server, args.model):
    if not os.path.exists(path):
        raise SystemExit(f"Not found: {path}")
os.makedirs("logs", exist_ok=True)
cmd = [args.server, "-m", args.model, "-ngl", "99", "-c", "32768", "-np", "1", "--jinja", "--port", str(args.port)]
print("Starting:", " ".join(cmd))
server = subprocess.Popen(cmd, stdout=open("logs/server_ask.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                          env=dict(os.environ, GGML_CUDA_DISABLE_GRAPHS="1"))
FIELDS = ["benchmark", "id", "mode", "correct", "expected", "model_answer", "finish", "thinking_words",
          "completion_tokens", "seconds", "question", "reply"]
rows, private = [], []
try:
    t0 = time.time()
    while health() != 200:
        if server.poll() is not None:
            raise SystemExit("llama-server stopped while loading. See logs/server_ask.log.")
        time.sleep(2)
    print(f"Server ready after {time.time() - t0:.0f} s\n")
    for mode in modes:
        for q in questions:
            try:
                content, reasoning, finish, usage, secs = ask(q["prompt"], mode)
            except Exception as e:
                content, reasoning, finish, usage, secs = "", "", f"error: {e}", {}, 0
            given = extract(q, content)
            ok = check(q, given, content)
            row = dict(benchmark=q["benchmark"], id=q["id"], mode=mode, correct=ok, expected=q["answer"],
                       model_answer=given[:200], finish=finish, thinking_words=len(reasoning.split()),
                       completion_tokens=usage.get("completion_tokens", ""), seconds=round(secs, 1),
                       question="" if q["private"] else q["question"][:500],
                       reply="" if q["private"] else " ".join(content.split())[:1500])
            if q["private"]:
                row["expected"] = q["answer"] if q["kind"] == "mc" else "(hidden)"
                row["model_answer"] = given[:200] if q["kind"] == "mc" else "(hidden)"
            rows.append(row)
            if q["private"]:  # full answers for checking by hand, kept out of the repo
                private.append(dict(row, expected=q["answer"], model_answer=given[:500], question=q["question"],
                                    reply=" ".join(content.split())))
            mark = {1: "right", 0: "wrong", "": "check"}[ok]
            print(f"[{mode:<8}] {q['benchmark']:<21} {mark:<5} {secs:6.1f} s  "
                  f"{row['completion_tokens']!s:>6} tokens  answer: {row['model_answer'][:40]}")
finally:
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
    print("Stopped llama-server")

os.makedirs("results", exist_ok=True)
out = "results/model_answers.csv"
keep = []
if os.path.exists(out):  # re-running some modes or benchmarks replaces just those rows
    done = {(r["benchmark"], r["id"], r["mode"]) for r in rows}
    with open(out, newline="", encoding="utf-8") as f:
        keep = [r for r in csv.DictReader(f) if (r["benchmark"], r["id"], r["mode"]) not in done]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(keep + rows)
print(f"\nSaved {out}")
if private:
    with open("questions/private_answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(private)
    print("Saved questions/private_answers.csv (GPQA and Humanity's Last Exam, not for the repo)")
for mode in modes:
    r = [x for x in rows if x["mode"] == mode]
    right = sum(1 for x in r if x["correct"] == 1)
    unsure = sum(1 for x in r if x["correct"] == "")
    print(f"{mode:<9} {right}/{len(r)} right" + (f", {unsure} to check by hand" if unsure else ""))
