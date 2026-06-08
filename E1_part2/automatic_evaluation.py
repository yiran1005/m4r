"""
automatic_evaluation.py — Compute J1 accuracy against StatQA ground truth.

For each (question_id, method) record:
  expected = APPLICABLE  if method in ground_truth_methods else NOT_APPLICABLE
  j1 = "correct" if record.verdict == expected else "wrong"

Outputs a CSV with one row per record, plus aggregate stats:
  - overall J1 accuracy
  - per-task-category accuracy
  - per-method accuracy
  - results for the high-frequency erroneous-retention methods, separately.

PARSE_ERROR records are reported but excluded from accuracy denominators
(printed warning if their share is non-trivial).

"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import config



def evaluate(parsed_path: Path, csv_out_path: Path) -> dict:
    records = []
    with parsed_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            expected = ("APPLICABLE"
                        if r["method"] in set(r["ground_truth_methods"])
                        else "NOT_APPLICABLE")
            r["expected"] = expected
            if r["verdict"] == "PARSE_ERROR":
                r["j1"] = "parse_error"
            else:
                r["j1"] = "correct" if r["verdict"] == expected else "wrong"
            records.append(r)

    # Write per-record CSV.
    fieldnames = ["question_id", "task", "difficulty", "method",
                  "verdict", "expected", "j1", "parse_reason",
                  "ground_truth_methods"]
    with csv_out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    return _summarise(records)


def _accuracy(records: list[dict]) -> tuple[float, int, int]:
    """Returns (acc, n_correct, n_evaluable). Excludes parse errors."""
    scored = [r for r in records if r["j1"] in ("correct", "wrong")]
    if not scored:
        return 0.0, 0, 0
    n_correct = sum(1 for r in scored if r["j1"] == "correct")
    return n_correct / len(scored), n_correct, len(scored)


def _summarise(records: list[dict]) -> dict:
    n_total = len(records)
    n_parse_err = sum(1 for r in records if r["j1"] == "parse_error")

    acc, n_corr, n_eval = _accuracy(records)

    print("=" * 70)
    print(f"Total records:  {n_total}")
    print(f"Parse errors:   {n_parse_err} ({n_parse_err/n_total*100:.1f}%)")
    print(f"Evaluable:      {n_eval}")
    print(f"Overall J1 acc: {acc*100:.1f}%  ({n_corr}/{n_eval})")
    if acc < 0.60:
        print(">>> C2 TRIGGERED: H1 weakened, bottleneck appears in Judgement.")
    elif acc < 0.75:
        print(">>> Partial support for H1 (60% ≤ J1 < 75%).")
    else:
        print(">>> Premise of H1 holds (J1 ≥ 75%). Proceed to Exp 2.")

    # Per-task breakdown.
    by_task: dict[str, list] = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    print("\nPer-task J1 accuracy:")
    for task, recs in sorted(by_task.items()):
        a, c, n = _accuracy(recs)
        print(f"  {task:32s} {a*100:5.1f}%  ({c}/{n})")

    # Per-method breakdown — sorted from worst to best.
    by_method: dict[str, list] = defaultdict(list)
    for r in records:
        by_method[r["method"]].append(r)
    rows = []
    for m, recs in by_method.items():
        a, c, n = _accuracy(recs)
        rows.append((m, a, c, n))
    rows.sort(key=lambda x: x[1])
    print("\nPer-method J1 accuracy (worst to best):")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.PARSED_JUDGEMENTS_PATH)
    parser.add_argument("--output", type=Path, default=config.J1_RESULTS_PATH)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Parsed file not found: {args.input}. "
                         "Run parse_outputs.py first.")
    evaluate(args.input, args.output)
    print(f"\nPer-record results written to {args.output}")


if __name__ == "__main__":
    main()