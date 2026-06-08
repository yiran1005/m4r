"""
prepare_annotation.py — Build the CSV that human annotators fill in.
====================================================================

Reads raw_responses.jsonl and emits a CSV with one row per method, plus
empty D1-D5 columns ready for annotation per the framework in Section 5.1.2:

    D1  Data type requirement      correct | partial | wrong | missing
    D2  Sample size requirement    correct | partial | wrong | missing | N/A
    D3  Statistical assumptions    correct | partial | wrong | missing
    D4  Exclusion conditions       correct | partial | wrong | missing
    D5  Discriminative sufficiency sufficient | insufficient | over-broad

"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import config


# --------------------------------------------------------------------------- #
# Selecting one representative response per method
# --------------------------------------------------------------------------- #
def select_representative_run(runs: list[dict]) -> dict:
    """Pick the run we want annotators to read for this method.

    With NUM_RUNS_PER_METHOD == 1 (the default given temp=0 decoding) this is
    just the single available run. With multiple stochastic runs, we pick the response whose token set has the highest
    average Jaccard similarity to the others (the run closest to the others
    in content). 
    """
    runs = [r for r in runs if r.get("response")]
    if not runs:
        return {}
    if len(runs) == 1:
        return runs[0]

    def tokenset(s: str) -> set[str]:
        return set(s.lower().split())

    token_sets = [tokenset(r["response"]) for r in runs]

    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        u = a | b
        return len(a & b) / len(u) if u else 0.0

    best_idx, best_score = 0, -1.0
    for i, ts_i in enumerate(token_sets):
        others = [ts_j for j, ts_j in enumerate(token_sets) if j != i]
        avg = sum(jaccard(ts_i, ts_j) for ts_j in others) / len(others)
        if avg > best_score:
            best_idx, best_score = i, avg
    chosen = dict(runs[best_idx])
    chosen["_consistency_score"] = round(best_score, 3)
    return chosen


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def group_runs_by_method(jsonl_path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            grouped[rec["method"]].append(rec)
    # Sort runs by run_idx so selection is deterministic.
    for m in grouped:
        grouped[m].sort(key=lambda r: r["run_idx"])
    return grouped


def write_picked_jsonl(picks: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in picks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_annotation_csv(picks: list[dict], path: Path) -> None:
    fieldnames = [
        "method",
        "response",
        # Annotator-filled columns
        "D1_data_type",          # correct / partial / wrong / missing
        "D2_sample_size",        # correct / partial / wrong / missing / N/A
        "D3_assumptions",        # correct / partial / wrong / missing
        "D4_exclusion",          # correct / partial / wrong / missing
        "D5_discriminative",     # sufficient / insufficient / over-broad
        "annotator_notes",
        # Provenance
        "run_idx",
        "seed",
        "model_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for rec in picks:
            writer.writerow({
                "method": rec.get("method", ""),
                "response": rec.get("response", ""),
                "D1_data_type": "",
                "D2_sample_size": "",
                "D3_assumptions": "",
                "D4_exclusion": "",
                "D5_discriminative": "",
                "annotator_notes": "",
                "run_idx": rec.get("run_idx", ""),
                "seed": rec.get("seed", ""),
                "model_id": rec.get("model_id", ""),
            })


def main():
    raw_path = config.RAW_RESPONSES_PATH
    if not raw_path.exists():
        raise SystemExit(
            f"No raw responses at {raw_path}. Run run_inference.py first."
        )

    grouped = group_runs_by_method(raw_path)
    print(f"Found responses for {len(grouped)} methods.")

    picks = [select_representative_run(runs) for runs in grouped.values()]
    picks = [p for p in picks if p]                # drop empties

    write_picked_jsonl(picks, config.CONSISTENCY_PICKED_PATH)
    write_annotation_csv(picks, config.ANNOTATION_CSV_PATH)

    print(f"Wrote picked responses to {config.CONSISTENCY_PICKED_PATH}")
    print(f"Wrote annotation CSV to   {config.ANNOTATION_CSV_PATH}")
    print(f"Methods in annotation CSV: {len(picks)}")


if __name__ == "__main__":
    main()