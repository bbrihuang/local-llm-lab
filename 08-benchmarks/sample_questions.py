"""Pick 3 real questions from each of the 10 benchmarks and save them to questions/questions.json.

Run from the 08-benchmarks folder:  ..\\.venv\\Scripts\\python sample_questions.py

Questions come from each benchmark's official dataset on Hugging Face (SimpleQA from OpenAI's
file), picked at random with a fixed seed, so every run picks the same ones. RULER is made of
generated tests, so its 3 are generated here the way RULER makes them.

GPQA and Humanity's Last Exam are "gated": you have to accept their terms on Hugging Face first,
and their authors ask that questions not be posted online, so they're kept out of the repo
(questions/ is gitignored, and the results files leave out their text).
"""
import csv
import io
import json
import os
import random
import re
import urllib.request

from datasets import load_dataset

os.chdir(os.path.dirname(os.path.abspath(__file__)))
N = 3
SEED = 8
LETTERS = "ABCDEFGHIJ"
MC_INSTRUCTION = ('Answer the following multiple choice question. Think step by step, then finish '
                  'your reply with "The answer is (X)", where X is the letter of the correct option.')
SHORT_INSTRUCTION = ('Answer the following question. Think step by step if you need to, then finish '
                     'your reply with "Final answer: " followed by just the answer.')


def mc(bench, qid, question, options, correct_index, private=False, shuffle=False, rng=None):
    options = [str(o).strip() for o in options]
    if shuffle:  # GPQA lists the right answer first in the dataset, so mix the options up
        order = list(range(len(options)))
        rng.shuffle(order)
        correct_index = order.index(correct_index)
        options = [options[i] for i in order]
    prompt = "\n".join([MC_INSTRUCTION, "", question.strip(), ""] +
                       [f"({LETTERS[i]}) {o}" for i, o in enumerate(options)])
    return dict(benchmark=bench, id=str(qid), kind="mc", private=private, question=question.strip(),
                options=options, answer=LETTERS[correct_index], prompt=prompt)


def short(bench, qid, question, answer, private=False):
    return dict(benchmark=bench, id=str(qid), kind="short", private=private, question=question.strip(),
                answer=str(answer).strip(), prompt=SHORT_INSTRUCTION + "\n\n" + question.strip())


def pick(rows, rng, n=N):
    return rng.sample(range(len(rows)), n)


def mmlu(rng):
    ds = load_dataset("cais/mmlu", "all", split="test")
    return [mc("MMLU", f"{ds[i]['subject']}/{i}", ds[i]["question"], ds[i]["choices"], ds[i]["answer"])
            for i in pick(ds, rng)]


def mmlu_pro(rng):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    return [mc("MMLU-Pro", f"{ds[i]['category']}/{ds[i]['question_id']}", ds[i]["question"],
               ds[i]["options"], ds[i]["answer_index"]) for i in pick(ds, rng)]


def gpqa(rng):
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out = []
    for i in pick(ds, rng):
        r = ds[i]
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        out.append(mc("GPQA-Diamond", r.get("Record ID", i), r["Question"], opts, 0,
                      private=True, shuffle=True, rng=rng))
    return out


def hle(rng):
    ds = load_dataset("cais/hle", split="test")
    text_only = [i for i, img in enumerate(ds["image"]) if not img]
    out = []
    for i in rng.sample(text_only, N):
        r = ds[i]
        q = short("Humanity's Last Exam", r["id"], r["question"], r["answer"], private=True)
        q["answer_type"] = r["answer_type"]  # exactMatch or multipleChoice (options are in the question)
        q["category"] = r.get("category", "")
        out.append(q)
    return out


def arc(rng):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    return [mc("ARC-Challenge", ds[i]["id"], ds[i]["question"], ds[i]["choices"]["text"],
               ds[i]["choices"]["label"].index(ds[i]["answerKey"])) for i in pick(ds, rng)]


def hellaswag(rng):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    out = []
    for i in pick(ds, rng):
        r = ds[i]
        ctx = re.sub(r"\[.*?\]", "", r["ctx"].replace(" [title]", ". ")).replace("  ", " ").strip()
        q = f"Which ending is the most likely continuation?\n\n{r['activity_label']}: {ctx} ..."
        endings = [re.sub(r"\[.*?\]", "", e).strip() for e in r["endings"]]
        out.append(mc("HellaSwag", r["ind"], q, endings, int(r["label"])))
    return out


def winogrande(rng):
    try:
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    except Exception:
        # The original repo uses a loading script, which newer versions of datasets refuse to run.
        # Hugging Face keeps an automatic parquet copy of it, so read that instead.
        ds = load_dataset("parquet", split="train", data_files="hf://datasets/allenai/winogrande@"
                          "refs%2Fconvert%2Fparquet/winogrande_xl/validation/*.parquet")
    # The parquet copy has no question IDs, so the row number stands in for one.
    return [mc("WinoGrande", ds[i].get("qID", f"validation/{i}"), f"Which option fits best in the blank (_)?\n\n{ds[i]['sentence']}",
               [ds[i]["option1"], ds[i]["option2"]], int(ds[i]["answer"]) - 1) for i in pick(ds, rng)]


def simpleqa(rng):
    url = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
    with urllib.request.urlopen(url, timeout=60) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))
    return [short("SimpleQA", i, rows[i]["problem"], rows[i]["answer"]) for i in pick(rows, rng)]


def ifeval(rng):
    ds = load_dataset("google/IFEval", split="train")
    out = []
    for i in pick(ds, rng):
        r = ds[i]
        out.append(dict(benchmark="IFEval", id=str(r["key"]), kind="ifeval", private=False,
                        question=r["prompt"], prompt=r["prompt"], answer="follows every instruction",
                        instruction_id_list=r["instruction_id_list"], kwargs=r["kwargs"]))
    return out


# RULER's "noise" haystack: the same filler sentences repeated, with facts hidden inside.
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
RULER_WORDS = ["harbor", "violet", "lantern", "granite", "meadow", "cobalt", "saffron", "willow"]


def haystack(rng, inserts, repeats=380):  # about 8,000 tokens
    chunks = [FILLER] * repeats
    for sentence in inserts:
        chunks.insert(rng.randrange(len(chunks) + 1), sentence + " ")
    return "".join(chunks)


def ruler(rng):
    out = []
    # 1. Single needle
    word, num = rng.choice(RULER_WORDS), rng.randrange(1000000, 9999999)
    text = haystack(rng, [f"One of the special magic numbers for {word} is: {num}."])
    q = f"What is the special magic number for {word} mentioned in the text above?"
    out.append(dict(benchmark="RULER", id="niah_single", kind="ruler", private=False, question=q,
                    answer=str(num), prompt=f"{text}\n\n{q} Finish your reply with \"Final answer: \" followed by the number."))
    # 2. Several needles, ask for one of them
    words = rng.sample(RULER_WORDS, 4)
    nums = [rng.randrange(1000000, 9999999) for _ in words]
    text = haystack(rng, [f"One of the special magic numbers for {w} is: {n}." for w, n in zip(words, nums)])
    q = f"What is the special magic number for {words[2]} mentioned in the text above?"
    out.append(dict(benchmark="RULER", id="niah_multikey", kind="ruler", private=False, question=q,
                    answer=str(nums[2]), prompt=f"{text}\n\n{q} Finish your reply with \"Final answer: \" followed by the number."))
    # 3. Variable tracking: follow a chain of assignments spread through the text
    value = rng.randrange(10000, 99999)
    names = ["".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)) for _ in range(4)]
    chain = [f"VAR {names[0]} = {value}."] + [f"VAR {names[k]} = VAR {names[k - 1]}." for k in range(1, 4)]
    decoy_value = rng.randrange(10000, 99999)
    decoys = ["".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)) for _ in range(2)]
    chain += [f"VAR {decoys[0]} = {decoy_value}.", f"VAR {decoys[1]} = VAR {decoys[0]}."]
    # keep the chain in order, but spread through the haystack
    chunks = [FILLER] * 380
    spots = sorted(rng.sample(range(len(chunks)), len(chain)))
    for k, s in sorted(zip(spots, chain), reverse=True):
        chunks.insert(k, s + " ")
    text = "".join(chunks)
    q = f"Find every variable that is assigned the value {value}, directly or through other variables."
    out.append(dict(benchmark="RULER", id="variable_tracking", kind="ruler", private=False, question=q,
                    answer=" ".join(names), prompt=f"{text}\n\n{q} Finish your reply with \"Final answer: \" followed by the variable names, separated by spaces."))
    return out


SOURCES = [("MMLU", mmlu), ("MMLU-Pro", mmlu_pro), ("GPQA-Diamond", gpqa), ("Humanity's Last Exam", hle),
           ("ARC-Challenge", arc), ("HellaSwag", hellaswag), ("WinoGrande", winogrande),
           ("SimpleQA", simpleqa), ("IFEval", ifeval), ("RULER", ruler)]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help='just these benchmarks, added to the saved file, e.g. "WinoGrande,GPQA-Diamond"')
    only = [b.strip() for b in ap.parse_args().only.split(",") if b.strip()]
    questions, failed = [], []
    if only and os.path.exists("questions/questions.json"):
        with open("questions/questions.json", encoding="utf-8") as f:
            questions = [q for q in json.load(f) if q["benchmark"] not in only]
    for name, fn in SOURCES:
        if only and name not in only:
            continue
        rng = random.Random(f"{SEED}-{name}")
        try:
            got = fn(rng)
            questions += got
            print(f"{name:<22} {len(got)} questions")
        except Exception as e:
            failed.append(name)
            print(f"{name:<22} FAILED: {type(e).__name__}: {str(e)[:400]}")
    order = [n for n, _ in SOURCES]
    questions.sort(key=lambda q: order.index(q["benchmark"]))
    os.makedirs("questions", exist_ok=True)
    with open("questions/questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=1)
    print(f"\nSaved {len(questions)} questions to questions/questions.json")
    if failed:
        print("Not loaded:", ", ".join(failed))
        if {"GPQA-Diamond", "Humanity's Last Exam"} & set(failed):
            print("For GPQA and Humanity's Last Exam, accept the terms on their pages first:\n"
                  "  https://huggingface.co/datasets/Idavidrein/gpqa\n"
                  "  https://huggingface.co/datasets/cais/hle")
