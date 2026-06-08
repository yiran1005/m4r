"""
parse_turn1.py — Parse Turn 1 JSON into a clean verdict table for Turn 2.
========================================================================

For each Turn 1 record:
  1. Extract the JSON object (tolerant of markdown fences / surrounding text).
  2. Read the model's Step-3 task_type and the Step-4 per-method verdicts.
  3. Determine Step-3 correctness: does the model's task_type match the
     ground-truth task category? (Section 5.2.2.1: Step-3 errors are flagged
     and excluded from the main analysis.)
  4. Restrict the verdict map to the canonical candidate set for the GROUND
     TRUTH task category, so Turn 2 always sees a well-formed table even if
     the model named extra/odd methods. Methods the model omitted are recorded
     as missing.
  5. Strip reason / conditions_checked — only verdict labels survive into the
     clean table (Section 5.2.3).

Records that fail JSON parsing or lack verdicts are logged to
PARSE_FAILURE_LOG and excluded (Section 5.2.2.1 safety measure).

Output: turn1_parsed.jsonl, one record per successfully parsed question, with
the clean verdict_map that run_turn2.py consumes.

Usage
-----
    python parse_turn1.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import config
from classification_list import (
    candidates_for_task, normalise_task, CLASSIFICATION_LIST,
)

VALID_VERDICTS = {"APPLICABLE", "NOT_APPLICABLE"}


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response.

    Handles ```json fences and leading/trailing prose. Returns None if no
    parseable object is found.
    """
    if not text:
        return None
    # Strip markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        # Fall back to the largest brace-balanced span starting at the first '{'.
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    break
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def normalise_verdict(v: str) -> str | None:
    if not isinstance(v, str):
        return None
    key = v.strip().upper().replace(" ", "_").replace("-", "_")
    if key in VALID_VERDICTS:
        return key
    if key in {"NOT_APPLICABLE", "NOTAPPLICABLE", "INAPPLICABLE"}:
        return "NOT_APPLICABLE"
    if key == "APPLICABLE":
        return "APPLICABLE"
    return None


def build_clean_table(obj: dict, gt_task: str) -> tuple[dict[str, str], dict]:
    """Return (clean_verdict_map, diagnostics).

    The clean table is keyed by the canonical candidate set for the ground
    truth task category. For each candidate:
      - if the model gave a valid verdict, use it
      - if the model omitted it, mark NOT_APPLICABLE and record as omitted
        (a conservative default; recorded in diagnostics so SSC analysis can
        treat omissions explicitly if desired)
    Methods the model named that are NOT in the candidate set are recorded as
    'extra_methods' for diagnostics but do not enter the table.
    """
    raw_summary = obj.get("summary", obj)  # tolerate missing "summary" wrapper
    model_verdicts: dict[str, str] = {}
    extra_methods = []
    for method, payload in raw_summary.items():
        if method not in CLASSIFICATION_LIST:
            continue  # ignore non-method keys like "task_type"
        if isinstance(payload, dict):
            v = normalise_verdict(payload.get("verdict", ""))
        else:
            v = normalise_verdict(str(payload))
        if v is None:
            continue
        model_verdicts[method] = v

    candidates = candidates_for_task(gt_task)
    clean: dict[str, str] = {}
    omitted = []
    for m in candidates:
        if m in model_verdicts:
            clean[m] = model_verdicts[m]
        else:
            clean[m] = "NOT_APPLICABLE"
            omitted.append(m)

    for m in model_verdicts:
        if m not in set(candidates):
            extra_methods.append(m)

    diagnostics = {
        "omitted_candidates": omitted,
        "extra_methods": extra_methods,
        "n_model_verdicts": len(model_verdicts),
    }
    return clean, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.TURN1_RAW_PATH)
    parser.add_argument("--output", type=Path, default=config.TURN1_PARSED_PATH)
    parser.add_argument("--failures", type=Path, default=config.PARSE_FAILURE_LOG)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Turn 1 raw not found: {args.input}")

    n_total = n_ok = n_fail = n_step3_err = 0
    with args.input.open("r", encoding="utf-8") as fin, \
         args.output.open("w", encoding="utf-8") as fout, \
         args.failures.open("w", encoding="utf-8") as ffail:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_total += 1
            qid = rec["question_id"]
            gt_task = rec["task"]

            obj = extract_json(rec.get("response", ""))
            if obj is None:
                n_fail += 1
                ffail.write(json.dumps({
                    "question_id": qid, "reason": "json_parse_failed",
                    "response_head": rec.get("response", "")[:300],
                }, ensure_ascii=False) + "\n")
                continue

            # Step-3 correctness: model's task_type vs ground truth task.
            model_task_raw = obj.get("task_type", "")
            model_task = normalise_task(str(model_task_raw))
            step3_correct = (model_task == gt_task)
            if not step3_correct:
                n_step3_err += 1

            clean, diag = build_clean_table(obj, gt_task)

            if not clean:
                n_fail += 1
                ffail.write(json.dumps({
                    "question_id": qid, "reason": "no_valid_verdicts",
                    "response_head": rec.get("response", "")[:300],
                }, ensure_ascii=False) + "\n")
                continue

            n_ok += 1
            # APPLICABLE-set size + recall against ground truth (Section 5.2.4:
            # mandatory auxiliary quantities to interpret SSC).
            applicable_set = {m for m, v in clean.items() if v == "APPLICABLE"}
            gt_set = set(rec["ground_truth_methods"])
            recall = (len(applicable_set & gt_set) / len(gt_set)
                      if gt_set else None)

            fout.write(json.dumps({
                "question_id": qid,
                "task": gt_task,
                "difficulty": rec["difficulty"],
                "question": rec["question"],
                "column_info": rec["column_info"],
                "ground_truth_methods": rec["ground_truth_methods"],
                "model_task_type": model_task_raw,
                "model_task_normalised": model_task,
                "step3_correct": step3_correct,
                "verdict_map": clean,
                "applicable_set": sorted(applicable_set),
                "applicable_set_size": len(applicable_set),
                "gt_recall": recall,
                "diagnostics": diag,
            }, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"Turn 1 records:        {n_total}")
    print(f"Parsed OK:             {n_ok}")
    print(f"Parse failures:        {n_fail}  (logged to {args.failures})")
    print(f"Step-3 task errors:    {n_step3_err}  "
          f"(flagged; exclude from main SSC analysis)")
    print("=" * 60)
    if n_fail:
        rate = n_fail / n_total * 100
        print(f"Parse failure rate: {rate:.1f}%")


if __name__ == "__main__":
    main()