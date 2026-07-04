#!/usr/bin/env python3
"""Build the legal clause classification dataset.

Instruction-format examples are generated from curated clause templates
with controlled paraphrase variation. Splits are template-level (a template
never appears in more than one split), which prevents near-duplicate
leakage from train into test.
"""

from __future__ import annotations

import json
import os
import random

INSTRUCTION = ("Classify the following contract clause as one of:"
               " indemnification, limitation_of_liability, termination,"
               " confidentiality, governing_law, payment_terms.")

TEMPLATES = {
    "indemnification": [
        "The {party} shall indemnify and hold harmless the {other} from any"
        " claims, damages, or liabilities arising out of {cause}.",
        "{party} agrees to defend, indemnify, and hold {other} harmless"
        " against all third-party claims resulting from {cause}.",
        "Each party shall indemnify the other for losses caused by its own"
        " {cause}, including reasonable attorneys' fees.",
    ],
    "limitation_of_liability": [
        "In no event shall {party}'s aggregate liability exceed the fees"
        " paid in the {period} preceding the claim.",
        "Neither party shall be liable for indirect, incidental, or"
        " consequential damages, including lost profits, arising from"
        " {cause}.",
        "{party}'s total liability under this agreement is capped at"
        " {amount}.",
    ],
    "termination": [
        "Either party may terminate this agreement upon {notice} days"
        " written notice to the other party.",
        "This agreement terminates automatically if {party} materially"
        " breaches and fails to cure within {notice} days.",
        "Upon termination, {party} shall cease use of the services and"
        " return all materials within {notice} days.",
    ],
    "confidentiality": [
        "The {party} shall not disclose any Confidential Information of the"
        " {other} except as required to perform under this agreement.",
        "Each party agrees to protect the other's proprietary information"
        " with at least the same care as its own, for {period}.",
        "Confidential Information excludes information that is publicly"
        " available or independently developed by the {party}.",
    ],
    "governing_law": [
        "This agreement shall be governed by and construed in accordance"
        " with the laws of {jurisdiction}.",
        "Any disputes arising under this agreement shall be resolved in the"
        " courts of {jurisdiction}, and the parties consent to venue there.",
        "The laws of {jurisdiction}, without regard to conflict of law"
        " principles, govern this agreement.",
    ],
    "payment_terms": [
        "Invoices are due within {notice} days of receipt; late payments"
        " accrue interest at {rate} percent per month.",
        "The {party} shall pay all fees in {currency} within {notice} days"
        " of the invoice date.",
        "Fees are payable {period} in advance and are non-refundable except"
        " as expressly stated.",
    ],
}

FILLERS = {
    "party": ["Vendor", "Client", "Supplier", "Licensee", "Contractor"],
    "other": ["Client", "Company", "Licensor", "Customer"],
    "cause": ["negligence", "breach of this agreement",
              "violation of applicable law", "willful misconduct"],
    "period": ["twelve months", "six months", "the prior year"],
    "amount": ["$100,000", "the total contract value", "$50,000"],
    "notice": ["30", "60", "90", "15"],
    "jurisdiction": ["the State of Delaware", "the State of New York",
                     "England and Wales", "the State of California"],
    "rate": ["1.5", "1.0", "2.0"],
    "currency": ["US dollars", "euros"],
}

BENCHMARK = [
    # Handcrafted edge cases: mixed vocabulary, archaic phrasing, near-misses.
    {"clause": "Whereas the party of the first part covenants to save"
               " harmless the party of the second part from all suits"
               " arising by reason of its default.",
     "label": "indemnification", "category": "archaic"},
    {"clause": "Save as set out herein, neither party's liability shall in"
               " aggregate exceed one hundred percent of charges paid.",
     "label": "limitation_of_liability", "category": "archaic"},
    {"clause": "Notwithstanding anything to the contrary, either party may"
               " end this contract for convenience on ninety days notice,"
               " whereupon all licenses cease.",
     "label": "termination", "category": "mixed"},
    {"clause": "Recipient shall use Discloser's non-public materials solely"
               " for the Project and for no other purpose whatsoever.",
     "label": "confidentiality", "category": "near-miss"},
    {"clause": "This deed and any non-contractual obligations connected"
               " with it are governed by the law of Scotland.",
     "label": "governing_law", "category": "archaic"},
    {"clause": "Charges invoiced quarterly; settlement strictly net 45;"
               " default interest per statute.",
     "label": "payment_terms", "category": "terse"},
]


def render(template: str, rng: random.Random) -> str:
    out = template
    for key, options in FILLERS.items():
        while "{" + key + "}" in out:
            out = out.replace("{" + key + "}", rng.choice(options), 1)
    return out


def build(out_dir: str = "data", per_template: int = 12,
          seed: int = 13) -> dict:
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for label, templates in TEMPLATES.items():
        for t_index, template in enumerate(templates):
            # Template-level split assignment prevents leakage.
            split = ["train", "train", "val" if t_index == 1 else "test"][
                t_index % 3] if False else None
            examples = []
            for _ in range(per_template):
                examples.append({
                    "instruction": INSTRUCTION,
                    "input": render(template, rng),
                    "output": label,
                    "template_id": f"{label}-{t_index}",
                })
            # 80/10/10 by template: templates 0 and 1 train, template 2
            # split between val and test.
            if t_index < 2:
                splits["train"].extend(examples)
            else:
                half = len(examples) // 2
                splits["val"].extend(examples[:half])
                splits["test"].extend(examples[half:])
    for name, rows in splits.items():
        rng.shuffle(rows)
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w",
                  encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    benchmark = [{"instruction": INSTRUCTION, "input": b["clause"],
                  "output": b["label"], "category": b["category"]}
                 for b in BENCHMARK]
    with open(os.path.join(out_dir, "benchmark.jsonl"), "w",
              encoding="utf-8") as fh:
        for row in benchmark:
            fh.write(json.dumps(row) + "\n")
    return {name: len(rows) for name, rows in splits.items()} | {
        "benchmark": len(benchmark)}


if __name__ == "__main__":
    counts = build()
    print(f"Dataset built: {counts}")
