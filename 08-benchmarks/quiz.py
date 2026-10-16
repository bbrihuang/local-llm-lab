"""Try the sample questions yourself, then see the right answer and what Qwen3.5-4B said.

Run from the 08-benchmarks folder:  ..\\.venv\\Scripts\\python quiz.py
Options: --only MMLU,HellaSwag   just some benchmarks
         --model-answers         also show Qwen's answers (after ask_model.py has run)

Type your answer and press Enter, or just press Enter to skip. Type q to stop; answers so far
are saved to results/my_answers.csv. No looking things up: the point is to feel how hard each
benchmark is without help, like the model.
"""
import argparse
import csv
import json
import os
import textwrap

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--model-answers", action="store_true")
args = ap.parse_args()

questions = json.load(open("questions/questions.json", encoding="utf-8"))
if args.only:
    questions = [q for q in questions if q["benchmark"] in args.only.split(",")]
model = {}
if args.model_answers and os.path.exists("results/model_answers.csv"):
    for r in csv.DictReader(open("results/model_answers.csv", encoding="utf-8")):
        model[(r["benchmark"], r["id"], r["mode"])] = r
    if os.path.exists("questions/private_answers.csv"):
        for r in csv.DictReader(open("questions/private_answers.csv", encoding="utf-8")):
            model[(r["benchmark"], r["id"], r["mode"])] = r


def show(text, width=100):
    for para in text.split("\n"):
        print(textwrap.fill(para, width) if para else "")


mine = []
for n, q in enumerate(questions, 1):
    print("\n" + "=" * 100)
    print(f"{n}/{len(questions)}  {q['benchmark']}")
    print("=" * 100)
    if q["kind"] == "ruler":
        words = len(q["prompt"].split())
        print(f"The model gets about {words:,} words of filler (\"The grass is green. The sky is blue. ...\") "
              "with a few facts hidden in it, then this question:\n")
        show(q["question"])
        print("\n(You'd need the whole text to answer, so just press Enter to see the hidden facts.)")
    elif q["kind"] == "mc":
        show(q["question"])
        print()
        for i, o in enumerate(q["options"]):
            show(f"  ({'ABCDEFGHIJ'[i]}) {o}")
    else:
        show(q["question"])
        if q["kind"] == "ifeval":
            print("\n(IFEval checks whether a reply follows every instruction: "
                  + ", ".join(q["instruction_id_list"]) + ". You don't need to write the reply; "
                  "just say whether you'd find it easy, or press Enter.)")
    ans = input("\nYour answer: ").strip()
    if ans.lower() == "q":
        break
    if q["kind"] == "ruler":
        print("Hidden facts, in the order they appear:")
        for part in q["prompt"].split(". "):
            if ("magic number" in part and "mentioned" not in part) or part.startswith("VAR "):
                print("   ", part.strip().rstrip(".") + ".")
    print(f"Right answer: {q['answer']}")
    for mode in ("instant", "thinking"):
        r = model.get((q["benchmark"], q["id"], mode))
        if r:
            mark = {"1": "right", "0": "wrong"}.get(r["correct"], "check by hand")
            print(f"Qwen ({mode}): {r['model_answer'][:120]}  [{mark}]")
    mine.append(dict(benchmark=q["benchmark"], id=q["id"], expected=q["answer"] if not q["private"] or q["kind"] == "mc" else "(hidden)",
                     my_answer=ans if not q["private"] or q["kind"] == "mc" else ("(answered)" if ans else "")))

os.makedirs("results", exist_ok=True)
with open("results/my_answers.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["benchmark", "id", "expected", "my_answer"])
    w.writeheader()
    w.writerows(mine)
print(f"\nSaved results/my_answers.csv ({len(mine)} answers)")
