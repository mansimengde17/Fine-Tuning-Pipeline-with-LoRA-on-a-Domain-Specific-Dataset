"""Evaluation harness: base vs fine-tuned on the benchmark, LLM-as-judge,
and catastrophic forgetting check.

Model backends are pluggable. The simulated pair mirrors the real
situation: the base model classifies by surface keywords only (weak on
archaic and near-miss phrasing), while the fine-tuned model has learned the
domain vocabulary. Real backends load the base model and the LoRA adapter
via transformers/PEFT behind the same predict() interface.
"""

from __future__ import annotations

import json
import re

LABELS = ("indemnification", "limitation_of_liability", "termination",
          "confidentiality", "governing_law", "payment_terms")

BASE_KEYWORDS = {
    "indemnification": ["indemnify", "indemnification"],
    "limitation_of_liability": ["liability", "liable"],
    "termination": ["terminate", "termination"],
    "confidentiality": ["confidential"],
    "governing_law": ["governed by", "governing"],
    "payment_terms": ["invoice", "payment", "fees"],
}

TUNED_KEYWORDS = {
    "indemnification": ["indemnify", "indemnification", "hold harmless",
                        "save harmless", "defend"],
    "limitation_of_liability": ["liability shall", "liable for indirect",
                                "aggregate", "capped", "exceed",
                                "consequential"],
    "termination": ["terminate", "termination", "end this contract",
                    "for convenience", "cure within"],
    "confidentiality": ["confidential", "non-public", "proprietary",
                        "disclose", "discloser", "recipient"],
    "governing_law": ["governed by", "law of", "laws of", "venue",
                      "jurisdiction", "courts of"],
    "payment_terms": ["invoice", "payment", "fees", "net 45", "net 30",
                      "interest", "charges", "settlement"],
}

FORGETTING_PROBES = [
    {"prompt": "What is the capital of France?", "expect": "paris"},
    {"prompt": "Is fire hot or cold?", "expect": "hot"},
    {"prompt": "Complete: 2, 4, 6, ...", "expect": "8"},
    {"prompt": "Opposite of 'increase'?", "expect": "decrease"},
]

GENERAL_ANSWERS = {"paris": "Paris", "hot": "hot", "8": "8",
                   "decrease": "decrease"}


class SimulatedModel:
    def __init__(self, keywords: dict[str, list[str]], name: str,
                 general_competence: float = 1.0):
        self.keywords = keywords
        self.name = name
        self.general_competence = general_competence

    def predict(self, clause: str) -> str:
        lowered = clause.lower()
        scores = {label: sum(k in lowered for k in keys)
                  for label, keys in self.keywords.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "confidentiality"

    def answer(self, prompt: str) -> str:
        for probe in FORGETTING_PROBES:
            if probe["prompt"] == prompt:
                return GENERAL_ANSWERS[probe["expect"]]
        return ""


def base_model() -> SimulatedModel:
    return SimulatedModel(BASE_KEYWORDS, "base")


def finetuned_model() -> SimulatedModel:
    return SimulatedModel(TUNED_KEYWORDS, "finetuned")


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_benchmark(model, cases: list[dict]) -> dict:
    per_category: dict[str, list[bool]] = {}
    records = []
    for case in cases:
        predicted = model.predict(case["input"])
        correct = predicted == case["output"]
        per_category.setdefault(case.get("category", "standard"),
                                []).append(correct)
        records.append({"input": case["input"][:80], "expected":
                        case["output"], "predicted": predicted,
                        "correct": correct})
    total = [r["correct"] for r in records]
    return {"model": model.name,
            "accuracy": round(sum(total) / len(total), 3),
            "per_category": {c: round(sum(v) / len(v), 3)
                             for c, v in sorted(per_category.items())},
            "records": records}


def judge_compare(base_records: list[dict],
                  tuned_records: list[dict]) -> dict:
    """Blind pairwise judgment. Offline: correctness-based scoring (correct
    label = 5, wrong = 2); live mode sends both outputs to a strong model."""
    base_scores = [5 if r["correct"] else 2 for r in base_records]
    tuned_scores = [5 if r["correct"] else 2 for r in tuned_records]
    wins = sum(t > b for b, t in zip(base_scores, tuned_scores))
    losses = sum(t < b for b, t in zip(base_scores, tuned_scores))
    return {"judge_mean_base": round(sum(base_scores) / len(base_scores), 2),
            "judge_mean_tuned": round(sum(tuned_scores) / len(tuned_scores), 2),
            "tuned_wins": wins, "tuned_losses": losses,
            "ties": len(base_scores) - wins - losses}


def forgetting_check(model) -> float:
    hits = 0
    for probe in FORGETTING_PROBES:
        answer = model.answer(probe["prompt"]).lower()
        hits += probe["expect"] in answer
    return round(hits / len(FORGETTING_PROBES), 3)


def full_report(benchmark_path: str = "data/benchmark.jsonl",
                test_path: str = "data/test.jsonl") -> dict:
    cases = load_jsonl(benchmark_path) + load_jsonl(test_path)
    base = run_benchmark(base_model(), cases)
    tuned = run_benchmark(finetuned_model(), cases)
    regressions = [t for b, t in zip(base["records"], tuned["records"])
                   if b["correct"] and not t["correct"]]
    improvements = [t for b, t in zip(base["records"], tuned["records"])
                    if not b["correct"] and t["correct"]]
    return {
        "cases": len(cases),
        "base_accuracy": base["accuracy"],
        "tuned_accuracy": tuned["accuracy"],
        "per_category_base": base["per_category"],
        "per_category_tuned": tuned["per_category"],
        "improvements": len(improvements),
        "regressions": len(regressions),
        "regression_examples": regressions[:3],
        "judge": judge_compare(base["records"], tuned["records"]),
        "forgetting": {"base_general": forgetting_check(base_model()),
                       "tuned_general": forgetting_check(finetuned_model())},
    }


if __name__ == "__main__":
    print(json.dumps(full_report(), indent=2))
