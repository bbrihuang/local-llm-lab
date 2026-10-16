"""Prompts and question picking for the two harness tasks (arc_challenge_qwen, mmlu_pro_qwen).

Both tasks ask the same way: the question, lettered options, and an instruction to end with
"The answer is (X)". This works the same with thinking on or off, and it's how Qwen's own
MMLU-Pro prompt asks for the answer.
"""
import random

LETTERS = "ABCDEFGHIJ"
INSTRUCTION = ('Answer the following multiple choice question. Think step by step, then finish '
               'your reply with "The answer is (X)", where X is the letter of the correct option.')


def _prompt(question, options):
    lines = [INSTRUCTION, "", question.strip(), ""]
    lines += [f"({LETTERS[i]}) {str(o).strip()}" for i, o in enumerate(options)]
    return "\n".join(lines)


# ARC-Challenge: the first 200 questions of the test set.
def arc_docs(dataset):
    return dataset.select(range(min(200, len(dataset))))


def arc_text(doc):
    return _prompt(doc["question"], doc["choices"]["text"])


def arc_target(doc):
    return LETTERS[doc["choices"]["label"].index(doc["answerKey"])]


# MMLU-Pro: 15 random questions from each of its 14 subjects (210 in all), the same ones every run.
PER_SUBJECT = 15


def mmlu_pro_docs(dataset):
    by_subject = {}
    for i, cat in enumerate(dataset["category"]):
        by_subject.setdefault(cat, []).append(i)
    rng = random.Random(1234)
    keep = []
    for cat in sorted(by_subject):
        keep += rng.sample(by_subject[cat], min(PER_SUBJECT, len(by_subject[cat])))
    return dataset.select(sorted(keep))


def mmlu_pro_text(doc):
    return _prompt(doc["question"], doc["options"])


def mmlu_pro_target(doc):
    return doc["answer"]
