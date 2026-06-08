"""
copen_analysis.py — Parse C-Open output and compute Jaccard vs main task.
=========================================================================

For each C-Open response:
  1. Parse the JSON object {"applicable_methods": [...]} into a set.
  2. Pull the per-method APPLICABLE set from parsed_judgements.jsonl for the
     same question_id.
  3. Compute Jaccard similarity.

The outline decision rules:
  Jaccard < 0.6  → leading effect significant; Part 2 results need caveat.
  Jaccard ≥ 0.8  → leading effect negligible; Part 2 results trustworthy.

"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

import config
from classification_list import CLASSIFICATION_LIST

CLASSIFICATION_SET = set(CLASSIFICATION_LIST)


def parse_copen_response(response: str) -> tuple[set[str], str]:
    """Extract the set of method names. Falls back to substring matching if
    JSON parsing fails. Returns (method_set, parse_status)."""
    if not response.strip():
        return set(), "empty"

    # 1. Try to find a JSON object.
    m = re.search(r"\{.*?\"applicable_methods\".*?\}", response, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            methods = obj.get("applicable_methods", [])
            # Keep only methods that match the canonical list (case-sensitive).
            kept = set(methods) & CLASSIFICATION_SET
            return kept, "json_ok"
        except json.JSONDecodeError:
            pass

    # 2. Fallback: substring match against the classification list.
    found = {m for m in CLASSIFICATION_LIST if m in response}
    if found:
        return found, "substring_fallback"

    return set(), "no_methods_found"


def applicable_set_from_main(parsed_path: Path,
                             question_ids: set[int]) -> dict[int, set[str]]:
    """For each requested question_id, return the set of methods the main
    Part 2 task labelled APPLICABLE."""
    out: dict[int, set[str]] = {qid: set() for qid in question_ids}
    with parsed_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = int(r["question_id"])
            if qid in out and r["verdict"] == "APPLICABLE":
                out[qid].add(r["method"])
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copen", type=Path, default=config.COPEN_OUTPUT_PATH)
    parser.add_argument("--main-parsed", type=Path,
                        default=config.PARSED_JUDGEMENTS_PATH)
    parser.add_argument("--out", type=Path, default=config.COPEN_REPORT_PATH)
    args = parser.parse_args()

    if not args.copen.exists():
        raise SystemExit(f"C-Open output not found: {args.copen}")
    if not args.main_parsed.exists():
        raise SystemExit(f"Main parsed output not found: {args.main_parsed}")

    # Read C-Open responses and parse them.
    copen_sets: dict[int, set[str]] = {}
    parse_status: dict[int, str] = {}
    gt_methods: dict[int, set[str]] = {}
    with args.copen.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = int(r["question_id"])
            methods, status = parse_copen_response(r.get("response", ""))
            copen_sets[qid] = methods
            parse_status[qid] = status
            gt_methods[qid] = set(r["ground_truth_methods"])

    # Look up the main-task APPLICABLE set for the same questions.
    main_sets = applicable_set_from_main(args.main_parsed, set(copen_sets))

    # Compute Jaccard per question.
    rows = []
    jaccards = []
    for qid in sorted(copen_sets):
        c, m = copen_sets[qid], main_sets[qid]
        j = jaccard(c, m)
        jaccards.append(j)
        rows.append({
            "question_id": qid,
            "copen_n": len(c),
            "main_n": len(m),
            "intersection": len(c & m),
            "union": len(c | m),
            "jaccard": round(j, 3),
            "copen_set": "; ".join(sorted(c)),
            "main_set": "; ".join(sorted(m)),
            "parse_status": parse_status[qid],
        })

    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows: w.writerow(r)

    print("=" * 70)
    print(f"Questions analysed: {len(jaccards)}")
    mean_j = mean(jaccards) if jaccards else 0.0
    print(f"Mean Jaccard:       {mean_j:.3f}")
    print(f"Min / Max:          {min(jaccards):.3f} / {max(jaccards):.3f}")

    if mean_j < 0.6:
        print(">>> Leading effect SIGNIFICANT (mean Jaccard < 0.6). "
              "Part 2 results need careful interpretation.")
    elif mean_j >= 0.8:
        print(">>> Leading effect NEGLIGIBLE (mean Jaccard ≥ 0.8). "
              "Part 2 results trustworthy.")
    else:
        print(">>> Leading effect AMBIGUOUS (0.6 ≤ mean Jaccard < 0.8). "
              "Discuss caveats in the thesis.")

    # Flag any parse failures.
    bad = [qid for qid, st in parse_status.items() if st in ("empty", "no_methods_found")]
    if bad:
        print(f"\nWARNING: {len(bad)} C-Open responses could not be parsed: "
              f"{bad[:5]}{'...' if len(bad) > 5 else ''}")
    print(f"\nFull report written to {args.out}")
    print("=" * 70)


if __name__ == "__main__":
    main()