"""
prepare_annotation.py — Stratified sample for J2 / J3 / J4 human annotation.
============================================================================

Per outline 5.1.4: from the 1350 J1-evaluated records, stratify-sample 100
with j1=correct and 100 with j1=wrong, give them to annotators to fill in:

  J2: did the model cite specific data features?     yes / no
  J3: were the cited reasons themselves correct?     correct / partially / wrong
  J4: consistency with Part 1 general-rule knowledge consistent_correct
                                                     | consistent_wrong
                                                     | contradictory

Parse-error records are excluded from sampling — annotating "the model
returned garbage" is not informative.

Reproducibility: random seed comes from config.ANNOTATION_SAMPLE_SEED.

"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import config


def stratified_sample(parsed_path: Path, n_per_stratum: int,
                      seed: int) -> tuple[list[dict], list[dict]]:
    """Return (correct_sample, wrong_sample) lists of records.

    Each record carries question_id, method, verdict, expected, response,
    and the J1 label.
    """
    correct_pool, wrong_pool = [], []
    with parsed_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            expected = ("APPLICABLE"
                        if r["method"] in set(r["ground_truth_methods"])
                        else "NOT_APPLICABLE")
            if r["verdict"] == "PARSE_ERROR":
                continue
            j1 = "correct" if r["verdict"] == expected else "wrong"
            r["expected"] = expected
            r["j1"] = j1
            (correct_pool if j1 == "correct" else wrong_pool).append(r)

    rng = random.Random(seed)
    correct = rng.sample(correct_pool, min(n_per_stratum, len(correct_pool)))
    wrong   = rng.sample(wrong_pool,   min(n_per_stratum, len(wrong_pool)))
    return correct, wrong


def write_annotation_csv(records: list[dict], path: Path) -> None:
    """One row per record. J2/J3/J4 columns are empty for annotators."""
    fieldnames = [
        "question_id", "task", "method",
        "expected", "verdict", "j1",
        "response",
        # Annotator-filled
        "J2_cites_features",          # yes / no
        "J3_reasons_correct",         # correct / partially_correct / wrong
        "J4_part1_consistency",       # consistent_correct / consistent_wrong / contradictory
        "annotator_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in records:
            w.writerow({
                "question_id": r["question_id"],
                "task": r.get("task", ""),
                "method": r["method"],
                "expected": r["expected"],
                "verdict": r["verdict"],
                "j1": r["j1"],
                "response": r.get("response", ""),
                "J2_cites_features": "",
                "J3_reasons_correct": "",
                "J4_part1_consistency": "",
                "annotator_notes": "",
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.PARSED_JUDGEMENTS_PATH)
    parser.add_argument("--output", type=Path, default=config.ANNOTATION_CSV_PATH)
    parser.add_argument("--n", type=int, default=config.ANNOTATION_SAMPLE_PER_STRATUM)
    parser.add_argument("--seed", type=int, default=config.ANNOTATION_SAMPLE_SEED)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Parsed file not found: {args.input}")

    correct, wrong = stratified_sample(args.input, args.n, args.seed)
    print(f"correct stratum: requested {args.n}, got {len(correct)}")
    print(f"wrong stratum:   requested {args.n}, got {len(wrong)}")

    # Interleave so annotators alternate; otherwise they may calibrate on
    # an all-correct or all-wrong run and miss the contrast.
    combined = []
    for i in range(max(len(correct), len(wrong))):
        if i < len(correct):
            combined.append(correct[i])
        if i < len(wrong):
            combined.append(wrong[i])

    write_annotation_csv(combined, args.output)
    print(f"Wrote {len(combined)} rows to {args.output}")


if __name__ == "__main__":
    main()