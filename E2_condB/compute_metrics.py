"""
compute_metrics.py — Cond C metrics with NA-pair stratification.

Since the oracle sets step3_correct=True for all, every question is in the
main analysis (no exclusions).

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

HIGH_FREQ = config.HIGH_FREQ_ERROR_METHODS


# Parse Turn 2 selection (same logic as Cond A)
def parse_selection(response: str) -> tuple[set[str], str]:
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
    found = {x for x in CLASSIFICATION_LIST if x in response}
    return found, "substring_fallback" if found else "no_methods_found"



# Optional cross-reference for high-disconnect-risk subset
def load_cot_retention(path: Path) -> set[tuple[int, str]]:
    """Load (question_id, method) pairs that were erroneously retained in
    4.3.3's CoT analysis. Expected columns: question_id, method.

    Returns an empty set if the file is absent (analysis then skipped).
    """
    if not path.exists():
        return set()
    pairs = set()
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pairs.add((int(row["question_id"]), row["method"].strip()))
            except (KeyError, ValueError):
                continue
    return pairs


# Main computation
def compute(turn2_path: Path, out_csv: Path, na_csv: Path) -> None:
    records = []
    with turn2_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    cot_retention = load_cot_retention(config.CoT_RETENTION_CSV)
    have_cot = len(cot_retention) > 0

    rows = []
    pool = []          # (sample, method) level records
    na_pairs = []      # detail of every NOT_APPLICABLE pair
    for rec in records:
        selected, sel_status = parse_selection(rec["response"])
        gt = set(rec["ground_truth_methods"])
        candidates = set(candidates_for_task(rec["task"]))
        out_of_summary = selected - candidates

        n_app = n_not = n_agree = n_add_in = n_drop = 0
        for method, verdict in rec["verdict_map"].items():
            is_sel = method in selected
            pool.append({"qid": rec["question_id"], "method": method,
                         "verdict": verdict, "selected": is_sel})
            if verdict == "APPLICABLE":
                n_app += 1
                n_agree += int(is_sel)
                n_drop += int(not is_sel)
            else:
                n_not += 1
                n_agree += int(not is_sel)
                n_add_in += int(is_sel)
                # Record NA-pair detail (for stratification).
                high_risk = (rec["question_id"], method) in cot_retention
                na_pairs.append({
                    "question_id": rec["question_id"],
                    "task": rec["task"],
                    "method": method,
                    "selected": int(is_sel),
                    "high_disconnect_risk": int(high_risk),
                    "is_high_freq": int(method in HIGH_FREQ),
                })

        n_pairs = n_app + n_not
        # Compute applicable_set_size and gt_recall on the fly from verdict_map
        # + ground truth, so we don't depend on run_turn2.py passing them
        # through. For the oracle these are trivially |GT| and 1.0, but
        # computing them here keeps compute_metrics robust to either source.
        applicable_in_summary = {m for m, v in rec["verdict_map"].items()
                                 if v == "APPLICABLE"}
        gt_recall = (len(applicable_in_summary & gt) / len(gt)
                     if gt else None)
        rows.append({
            "question_id": rec["question_id"],
            "task": rec["task"],
            "ssc": n_agree / n_pairs if n_pairs else None,
            "add_rate_in": n_add_in / n_not if n_not else None,
            "drop_rate": n_drop / n_app if n_app else None,
            "add_out_count": len(out_of_summary),
            "selection_accuracy": int(selected == gt),
            "applicable_set_size": len(applicable_in_summary),
            "gt_recall": gt_recall,
            "selection_parse_status": sel_status,
        })

    # Write per-question CSV.
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write NA-pair detail CSV.
    if na_pairs:
        with na_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(na_pairs[0].keys()),
                               quoting=csv.QUOTE_ALL)
            w.writeheader()
            for r in na_pairs:
                w.writerow(r)

    _report(rows, pool, na_pairs, have_cot)


# Reporting
def _pooled_ssc(pairs):
    if not pairs:
        return None, 0
    agree = sum(1 for p in pairs
                if (p["verdict"] == "APPLICABLE" and p["selected"])
                or (p["verdict"] == "NOT_APPLICABLE" and not p["selected"]))
    return agree / len(pairs), len(pairs)


def _pooled_add_in(pairs):
    nots = [p for p in pairs if p["verdict"] == "NOT_APPLICABLE"]
    if not nots:
        return None, 0
    return sum(1 for p in nots if p["selected"]) / len(nots), len(nots)


def _pooled_drop(pairs):
    apps = [p for p in pairs if p["verdict"] == "APPLICABLE"]
    if not apps:
        return None, 0
    return sum(1 for p in apps if not p["selected"]) / len(apps), len(apps)


def _report(rows, pool, na_pairs, have_cot):
    print("=" * 72)
    print(f"Cond C — model: {config.MODEL_KEY}")
    print(f"Total questions: {len(rows)} (all in main analysis; oracle)")
    print("=" * 72)

    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return mean(vals) if vals else float("nan")

    print("\n--- OVERALL (per-question mean) ---")
    print(f"Mean SSC:            {avg('ssc'):.3f}")
    print(f"Mean Add Rate (in):  {avg('add_rate_in'):.3f}")
    print(f"Mean Drop Rate:      {avg('drop_rate'):.3f}   "
          "(>0 = over-exclude disconnect; Section 5.2.5)")
    print(f"Selection Accuracy:  {avg('selection_accuracy'):.3f}   "
          "(<1.0 strongly supports H1; Section 5.2.6)")
    total_out = sum(r["add_out_count"] for r in rows)
    print(f"Out-of-summary adds: {total_out}")

    # Pooled subgroup metrics.
    hi = [p for p in pool if p["method"] in HIGH_FREQ]
    lo = [p for p in pool if p["method"] not in HIGH_FREQ]
    print("\n--- POOLED (sample, method) SUBGROUP METRICS ---")
    for label, pairs in (("High-freq error methods", hi),
                         ("Other methods", lo),
                         ("ALL", pool)):
        ssc, n = _pooled_ssc(pairs)
        add_in, n_na = _pooled_add_in(pairs)
        drop, n_app = _pooled_drop(pairs)
        ssc_s = f"{ssc:.3f}" if ssc is not None else "n/a"
        add_s = f"{add_in:.3f}" if add_in is not None else "n/a"
        drop_s = f"{drop:.3f}" if drop is not None else "n/a"
        print(f"  {label:26s} SSC={ssc_s} (n={n:4d})  "
              f"Add(in)={add_s} (NA={n_na})  Drop={drop_s} (APP={n_app})")

    # Generalisation-test group (12 robust methods), if specified.
    if config.ROBUST_OTHER_METHODS:
        robust_set = set(config.ROBUST_OTHER_METHODS)
        rob = [p for p in pool if p["method"] in robust_set]
        ssc, n = _pooled_ssc(rob)
        ssc_s = f"{ssc:.3f}" if ssc is not None else "n/a"
        # How many of the listed robust methods actually appear in the pool?
        # Some (e.g. DS methods Range/Quartile/Mode) never enter S50 candidate
        # sets because S50 has no DS questions, so they
        # contribute zero pairs here.
        present = {p["method"] for p in rob}
        absent = robust_set - present
        print(f"\n  Generalisation group: {len(present)}/"
              f"{len(robust_set)} listed methods appear in S50 candidate sets")
        print(f"    Pooled SSC = {ssc_s} (n_pairs={n})")
        if absent:
            print(f"    Not present (no S50 candidate set): "
                  f"{sorted(absent)}")
        print("    SSC<1.0 -> disconnect is general; SSC~1.0 -> limited to "
              "high-freq methods (Section 5.2.5.1)")

    # Per-method NA-pair counts and Add Rate(in) for high-freq methods.
    print("\n--- HIGH-FREQ METHODS: NA pairs & Add Rate(in) ---")
    print("  (Section 5.2.7 expected NA counts: Bartlett 12, F-Test 12, "
          "MH 9, Fisher 5, AD 3, Pearson 1)")
    for m in HIGH_FREQ:
        nots = [p for p in pool
                if p["method"] == m and p["verdict"] == "NOT_APPLICABLE"]
        if not nots:
            print(f"    {m:42s} no NA pairs")
            continue
        added = sum(1 for p in nots if p["selected"])
        flag = "" if len(nots) >= 5 else "  (low power: pool only)"
        print(f"    {m:42s} Add(in)={added}/{len(nots)}={added/len(nots):.2f}"
              f"{flag}")

    # High-disconnect-risk subset
    print("\n--- HIGH-DISCONNECT-RISK SUBSET (Section 5.2.4) ---")
    if not have_cot:
        print("  4.3.3 cross-reference CSV not found "
              f"({config.CoT_RETENTION_CSV.name}).")
        print("  Reporting FULL-SET Add Rate(in) only (the upper bound).")
        nots = [p for p in na_pairs]
        added = sum(p["selected"] for p in nots)
        if nots:
            print(f"  Full set: {added}/{len(nots)} = {added/len(nots):.3f}")
    else:
        risk = [p for p in na_pairs if p["high_disconnect_risk"]]
        full = na_pairs
        r_add = sum(p["selected"] for p in risk)
        f_add = sum(p["selected"] for p in full)
        print(f"  High-risk subset: {r_add}/{len(risk)} = "
              f"{r_add/len(risk):.3f}" if risk else "  High-risk subset: empty")
        print(f"  Full set (upper): {f_add}/{len(full)} = "
              f"{f_add/len(full):.3f}")
        print("  Diagnostic conclusion should rest on the high-risk subset.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn2", type=Path, default=config.TURN2_RAW_PATH)
    parser.add_argument("--out", type=Path, default=config.METRICS_PATH)
    parser.add_argument("--na-out", type=Path, default=config.NA_PAIR_DETAIL_PATH)
    args = parser.parse_args()

    if not args.turn2.exists():
        raise SystemExit(f"Turn 2 output not found: {args.turn2}")
    compute(args.turn2, args.out, args.na_out)
    print(f"\nPer-question metrics: {args.out}")
    print(f"NA-pair detail:       {args.na_out}")


if __name__ == "__main__":
    main()