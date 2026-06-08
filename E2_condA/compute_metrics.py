"""
compute_metrics.py — SSC, Add/Drop rates, summary-size & recall (Cond A).
========================================================================

Joins Turn 2 selections with Turn 1 verdict tables and ground truth, then
computes the Section 5.2.5 metrics. All metrics are reported at two levels
(Section 5.2.5.1): overall, and for the 6 high-frequency error methods.

Metric definitions (all restricted to methods that appear in the summary,
i.e. the candidate set — per the pruning design):

  SSC  = fraction of (method) where Turn-2 selection agrees with summary verdict
         agreement: method selected  AND verdict==APPLICABLE, OR
                    method not selected AND verdict==NOT_APPLICABLE
  Add Rate (in-summary)  = of methods marked NOT_APPLICABLE, fraction selected
  Add Rate (out-of-summary) = methods selected that are NOT in the candidate set
  Drop Rate = of methods marked APPLICABLE, fraction NOT selected
  Selection Accuracy = exact match of selected set vs ground-truth method set

Auxiliary (Section 5.2.4, mandatory for interpreting SSC):
  applicable_set_size  = |APPLICABLE in summary|
  gt_recall            = |APPLICABLE ∩ GT| / |GT|

Step-3-incorrect questions are excluded from the main aggregates (Section
5.2.2.1) but counted separately.


"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

import config
from classification_list import candidates_for_task, CLASSIFICATION_LIST

HIGH_FREQ = set(config.HIGH_FREQ_ERROR_METHODS)


# --------------------------------------------------------------------------- #
# Parse Turn 2 selection
# --------------------------------------------------------------------------- #
def parse_selection(response: str) -> tuple[set[str], str]:
    """Extract selected_methods from Turn 2 output. Returns (set, status)."""
    if not response.strip():
        return set(), "empty"
    m = re.search(r"\{.*\}", response, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            methods = obj.get("selected_methods", [])
            kept = {x for x in methods if x in set(CLASSIFICATION_LIST)}
            return kept, "json_ok"
        except json.JSONDecodeError:
            pass
    # Fallback: substring match.
    found = {x for x in CLASSIFICATION_LIST if x in response}
    return found, "substring_fallback" if found else "no_methods_found"


# --------------------------------------------------------------------------- #
# Per-question metric computation
# --------------------------------------------------------------------------- #
def per_question_pairs(rec: dict, selected: set[str]):
    """Yield (method, verdict, selected_bool, in_candidate_set) for each
    candidate method in this question's summary."""
    verdict_map = rec["verdict_map"]
    for method, verdict in verdict_map.items():
        yield method, verdict, (method in selected), True


def compute(turn2_path: Path, out_csv: Path) -> None:
    records = []
    with turn2_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Per-question rows + global pair pools (for pooled subgroup metrics).
    rows = []
    # Pools of (verdict, selected) at the (sample, method) level.
    pool_all = []          # list of dicts: method, verdict, selected, qid
    for rec in records:
        selected, sel_status = parse_selection(rec["response"])
        gt = set(rec["ground_truth_methods"])
        candidates = set(candidates_for_task(rec["task"]))

        # out-of-summary additions: selected methods not in candidate set.
        out_of_summary = selected - candidates

        n_app = n_not = n_agree = n_add_in = n_drop = 0
        for method, verdict, is_sel, _ in per_question_pairs(rec, selected):
            pool_all.append({"qid": rec["question_id"], "method": method,
                             "verdict": verdict, "selected": is_sel})
            if verdict == "APPLICABLE":
                n_app += 1
                if is_sel:
                    n_agree += 1
                else:
                    n_drop += 1
            else:  # NOT_APPLICABLE
                n_not += 1
                if not is_sel:
                    n_agree += 1
                else:
                    n_add_in += 1

        n_pairs = n_app + n_not
        ssc = n_agree / n_pairs if n_pairs else None
        add_in = n_add_in / n_not if n_not else None
        drop = n_drop / n_app if n_app else None
        sel_acc = int(selected == gt)

        rows.append({
            "question_id": rec["question_id"],
            "task": rec["task"],
            "step3_correct": rec["step3_correct"],
            "ssc": ssc,
            "add_rate_in": add_in,
            "drop_rate": drop,
            "add_out_count": len(out_of_summary),
            "out_of_summary_methods": "; ".join(sorted(out_of_summary)),
            "selection_accuracy": sel_acc,
            "applicable_set_size": sum(1 for v in rec["verdict_map"].values()
                                       if v == "APPLICABLE"),
            "gt_recall": (len({m for m, v in rec["verdict_map"].items()
                               if v == "APPLICABLE"} & gt) / len(gt)
                          if gt else None),
            "selection_parse_status": sel_status,
        })

    # Write per-question CSV.
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    _report(rows, pool_all)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _pooled_ssc(pairs: list[dict]) -> tuple[float, int]:
    if not pairs:
        return None, 0
    agree = sum(1 for p in pairs
                if (p["verdict"] == "APPLICABLE" and p["selected"])
                or (p["verdict"] == "NOT_APPLICABLE" and not p["selected"]))
    return agree / len(pairs), len(pairs)


def _pooled_add_in(pairs: list[dict]) -> tuple[float, int]:
    nots = [p for p in pairs if p["verdict"] == "NOT_APPLICABLE"]
    if not nots:
        return None, 0
    added = sum(1 for p in nots if p["selected"])
    return added / len(nots), len(nots)


def _report(rows: list[dict], pool_all: list[dict]) -> None:
    main = [r for r in rows if r["step3_correct"]]
    excluded = [r for r in rows if not r["step3_correct"]]

    print("=" * 70)
    print(f"Total questions:        {len(rows)}")
    print(f"Step-3 correct (main):  {len(main)}")
    print(f"Step-3 errors (excl.):  {len(excluded)}")
    print("=" * 70)

    def avg(key, recs):
        vals = [r[key] for r in recs if r[key] is not None]
        return mean(vals) if vals else float("nan")

    print("\n--- MAIN ANALYSIS (Step-3 correct only) ---")
    print(f"Mean SSC:                {avg('ssc', main):.3f}")
    print(f"Mean Add Rate (in):      {avg('add_rate_in', main):.3f}")
    print(f"Mean Drop Rate:          {avg('drop_rate', main):.3f}")
    print(f"Selection Accuracy:      {avg('selection_accuracy', main):.3f}")
    print(f"Mean APPLICABLE-set size:{avg('applicable_set_size', main):.2f}")
    print(f"Mean GT recall:          {avg('gt_recall', main):.3f}")
    total_out = sum(r["add_out_count"] for r in main)
    print(f"Out-of-summary additions:{total_out} (across {len(main)} questions)")

    print("\n   Reminder (Section 5.2.4): interpret SSC together with "
          "APPLICABLE-set size and GT recall.\n   A narrow summary inflates "
          "SSC without proving absence of disconnect.")

    # Pooled subgroup metrics (high-freq vs others) on Step-3-correct questions.
    main_qids = {r["question_id"] for r in main}
    pool_main = [p for p in pool_all if p["qid"] in main_qids]
    hi = [p for p in pool_main if p["method"] in HIGH_FREQ]
    lo = [p for p in pool_main if p["method"] not in HIGH_FREQ]

    print("\n--- POOLED (sample, method) SUBGROUP METRICS [Step-3 correct] ---")
    for label, pairs in (("High-freq error methods", hi),
                         ("Other methods", lo),
                         ("ALL", pool_main)):
        ssc, n = _pooled_ssc(pairs)
        add_in, n_not = _pooled_add_in(pairs)
        ssc_s = f"{ssc:.3f}" if ssc is not None else "n/a"
        add_s = f"{add_in:.3f}" if add_in is not None else "n/a"
        print(f"  {label:26s} SSC={ssc_s} (n_pairs={n:4d})  "
              f"AddRate(in)={add_s} (n_NA={n_not})")

    # Per-method Add Rate (in) for the high-freq methods specifically.
    print("\n  High-freq Add Rate(in) per method "
          "(>0 = 'knew NOT_APPLICABLE, still selected'):")
    for m in config.HIGH_FREQ_ERROR_METHODS:
        nots = [p for p in pool_main
                if p["method"] == m and p["verdict"] == "NOT_APPLICABLE"]
        if not nots:
            print(f"    {m:42s} no NA pairs")
            continue
        added = sum(1 for p in nots if p["selected"])
        print(f"    {m:42s} {added}/{len(nots)} = {added/len(nots):.2f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn2", type=Path, default=config.TURN2_RAW_PATH)
    parser.add_argument("--out", type=Path, default=config.METRICS_PATH)
    args = parser.parse_args()

    if not args.turn2.exists():
        raise SystemExit(f"Turn 2 output not found: {args.turn2}")
    compute(args.turn2, args.out)
    print(f"\nPer-question metrics written to {args.out}")


if __name__ == "__main__":
    main()