"""
construct_oracle.py — Build the oracle verdict tables (no model needed).


For each S50 question:
  candidates = candidate methods of the GROUND-TRUTH task category
  verdict_map[m] = APPLICABLE      if m in ground_truth_methods
                 = NOT_APPLICABLE  otherwise

This is a pure set operation: (candidate set) minus (applicable set) = the
NOT_APPLICABLE set. 

The output format is intentionally identical to Cond A's turn1_parsed.jsonl,
so the shared run_turn2.py consumes it without modification. Differences:
  - step3_correct is always True (the oracle has no Step-3 stage; every
    question enters the main analysis).
  - applicable_set == ground_truth_methods by construction, so gt_recall == 1.0
    (the oracle summary is, by definition, a perfect-recall summary — this is
    exactly why Cond C avoids the narrowing confound that threatens Cond A).

Sanity check baked in: every ground-truth method MUST fall inside its task's
candidate set, else the oracle construction is ill-defined. We assert this and
report any violations (there should be none for S50 per the design).

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from classification_list import candidates_for_task
from data_loader import load_s50


def build_oracle_record(sample) -> tuple[dict | None, str | None]:
    """Return (record, error). error is non-None if a GT method is outside the
    task candidate set (which would make the oracle table ill-defined)."""
    candidates = candidates_for_task(sample.task)
    cand_set = set(candidates)
    gt_set = set(sample.ground_truth_methods)

    outside = gt_set - cand_set
    if outside:
        return None, (f"ground-truth methods outside candidate set for task "
                      f"{sample.task!r}: {sorted(outside)}")

    verdict_map = {m: ("APPLICABLE" if m in gt_set else "NOT_APPLICABLE")
                   for m in candidates}
    applicable_set = sorted(m for m in candidates if m in gt_set)

    record = {
        "question_id": sample.question_id,
        "task": sample.task,
        "difficulty": sample.difficulty,
        "question": sample.question,
        "column_info": sample.column_info,
        "ground_truth_methods": sample.ground_truth_methods,
        # Fields mirroring Cond A's turn1_parsed.jsonl so run_turn2.py works:
        "model_task_type": sample.task,          # oracle: task is given
        "model_task_normalised": sample.task,
        "step3_correct": True,                   # oracle has no Step-3 stage
        "verdict_map": verdict_map,
        "applicable_set": applicable_set,
        "applicable_set_size": len(applicable_set),
        "gt_recall": 1.0,                        # perfect by construction
        "diagnostics": {"source": "oracle_ground_truth"},
    }
    return record, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=config.ORACLE_PARSED_PATH)
    args = parser.parse_args()

    samples = load_s50()
    n_ok = 0
    n_err = 0
    na_total = 0
    with args.output.open("w", encoding="utf-8") as f:
        for s in samples:
            record, error = build_oracle_record(s)
            if error:
                n_err += 1
                print(f"  WARNING Q{s.question_id}: {error}")
                continue
            n_ok += 1
            na_total += sum(1 for v in record["verdict_map"].values()
                            if v == "NOT_APPLICABLE")
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"Oracle tables built:   {n_ok}/{len(samples)}")
    if n_err:
        print(f"Ill-defined (skipped): {n_err}  "
              "(GT method outside candidate set — investigate!)")
    print(f"Total NOT_APPLICABLE pairs across all questions: {na_total}")
    print(f"Output: {args.output}")
    print("=" * 60)
    print("Next: qsub run_turn2.pbs  (runs the shared Turn 2 on this oracle)")


if __name__ == "__main__":
    main()