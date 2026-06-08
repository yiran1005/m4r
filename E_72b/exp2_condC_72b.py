"""
exp2_condC_72b.py — Exp 2 Cond C (Oracle Summary) on Qwen2.5-72B via API.
========================================================================

72B Layer-2 run: accuracy + automatic SSC (Section 5.0.4 / 5.2.4 / 5.2.7).
No Turn 1 — the verdict table is constructed deterministically from StatQA
ground truth, then the SAME Turn 2 as Cond A selects from it.

This is the key cross-capacity evidence: comparing Cond C SSC/accuracy across
7B / 14B / 72B tells us whether the disconnect has a capacity threshold.

Sample set: shared S50 (same questions as 14B Cond C).

Usage
-----
    export DASHSCOPE_API_KEY=sk-xxxx
    python exp2_condC_72b.py --dry-run
    python exp2_condC_72b.py                 # oracle -> Turn2 -> metrics
    python exp2_condC_72b.py --stage metrics # re-run metrics only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

import api_client
from classification_list import CLASSIFICATION_LIST, candidates_for_task
from data_loader import load_s50

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
S50_CSV = PROJECT_ROOT / "qwen2_5_14b_zero-shot-CoT_50_with_AE_component.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
ORACLE_PARSED = OUTPUT_DIR / "condC_72b_oracle.jsonl"
TURN2_RAW = OUTPUT_DIR / "condC_72b_turn2_raw.jsonl"
METRICS_CSV = OUTPUT_DIR / "condC_72b_metrics.csv"
NA_PAIR_CSV = OUTPUT_DIR / "condC_72b_na_pairs.csv"
TURN2_MAX_TOKENS = 512

HIGH_FREQ = [
    "Anderson-Darling Test", "Fisher Exact Test", "Mantel-Haenszel Test",
    "Pearson Correlation Coefficient", "Bartlett Test", "F-Test for Variance",
]
# 12 judgement-robust methods (generalisation-test group, Section 5.2.5.1).
ROBUST_OTHER = [
    "Partial Correlation Coefficient", "Chi-square Independence Test",
    "Kolmogorov-Smirnov Test for Normality",
    "Kolmogorov-Smirnov Test for Uniform distribution",
    "Kolmogorov-Smirnov Test for Gamma distribution",
    "Kolmogorov-Smirnov Test for Exponential distribution",
    "Lilliefors Test", "Kendall Correlation Coefficient", "Levene Test",
    "Range", "Quartile", "Mode",
]

# Same Turn 2 prompt as Cond A (Section 5.2.1.1 requires identical Turn 2).
TURN2_SYSTEM = (
    "You are a careful statistical assistant making a final method selection "
    "based strictly on a provided applicability table. You do not re-evaluate."
)
TURN2_USER = """An applicability analysis has been completed for this problem. The results are:

{verdict_table}

Problem: {question}

Column information:
{column_info}

Based STRICTLY on the table above, output the final answer as a JSON object:
{{
  "selected_methods": [ ... ]
}}

Rules:
- Include ONLY methods marked APPLICABLE in the table.
- Do not re-evaluate any method.
- Do not add methods not marked APPLICABLE.
- Do not omit methods marked APPLICABLE.
Output only the JSON object."""


# --------------------------------------------------------------------------- #
# Stage 1: construct oracle (no API)
# --------------------------------------------------------------------------- #
def construct_oracle():
    samples = load_s50(S50_CSV)
    n_ok = n_err = 0
    with ORACLE_PARSED.open("w") as f:
        for s in samples:
            candidates = candidates_for_task(s.task)
            gt = set(s.ground_truth_methods)
            outside = gt - set(candidates)
            if outside:
                n_err += 1
                print(f"  WARNING Q{s.question_id}: GT outside candidates: {outside}")
                continue
            verdict_map = {m: ("APPLICABLE" if m in gt else "NOT_APPLICABLE")
                           for m in candidates}
            n_ok += 1
            f.write(json.dumps({
                "question_id": s.question_id, "task": s.task,
                "question": s.question, "column_info": s.column_info,
                "ground_truth_methods": s.ground_truth_methods,
                "step3_correct": True,
                "verdict_map": verdict_map,
                "applicable_set": sorted(m for m in candidates if m in gt),
            }, ensure_ascii=False) + "\n")
    print(f"Oracle built: {n_ok}/{len(samples)}  (ill-defined: {n_err})")


# --------------------------------------------------------------------------- #
# Stage 2: Turn 2
# --------------------------------------------------------------------------- #
def render_table(verdict_map):
    return "\n".join(f"- {m}: {v}" for m, v in verdict_map.items())


def build_turn2_messages(task: dict) -> list[dict]:
    return [
        {"role": "system", "content": TURN2_SYSTEM},
        {"role": "user", "content": TURN2_USER.format(
            verdict_table=render_table(task["verdict_map"]),
            question=task["question"], column_info=task["column_info"],
        )},
    ]


def turn2_record(task: dict, text: str, error: str | None) -> dict:
    return {
        "question_id": task["question_id"], "task": task["task"],
        "step3_correct": True, "verdict_map": task["verdict_map"],
        "ground_truth_methods": task["ground_truth_methods"],
        "response": text, "error": error,
    }


def run_turn2(dry_run=False):
    oracle = [json.loads(l) for l in ORACLE_PARSED.open() if l.strip()]
    if dry_run:
        print(build_turn2_messages(oracle[0])[-1]["content"])
        print("\n(dry run — no API calls)")
        return
    api_client.run_batch(oracle, build_turn2_messages, turn2_record,
                         TURN2_RAW, id_field="question_id",
                         max_tokens=TURN2_MAX_TOKENS)


# --------------------------------------------------------------------------- #
# Stage 3: metrics
# --------------------------------------------------------------------------- #
def parse_selection(response: str):
    if not response.strip():
        return set()
    m = re.search(r"\{.*\}", response, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return {x for x in obj.get("selected_methods", [])
                    if x in set(CLASSIFICATION_LIST)}
        except json.JSONDecodeError:
            pass
    return {x for x in CLASSIFICATION_LIST if x in response}


def compute_metrics():
    records = [json.loads(l) for l in TURN2_RAW.open() if l.strip()]
    rows = []
    pool = []
    na_pairs = []
    for rec in records:
        selected = parse_selection(rec["response"])
        gt = set(rec["ground_truth_methods"])
        candidates = set(candidates_for_task(rec["task"]))
        out_of_summary = selected - candidates
        n_app = n_not = n_agree = n_add_in = n_drop = 0
        for method, verdict in rec["verdict_map"].items():
            is_sel = method in selected
            pool.append({"method": method, "verdict": verdict, "selected": is_sel})
            if verdict == "APPLICABLE":
                n_app += 1
                n_agree += int(is_sel); n_drop += int(not is_sel)
            else:
                n_not += 1
                n_agree += int(not is_sel); n_add_in += int(is_sel)
                na_pairs.append({"question_id": rec["question_id"],
                                 "method": method, "selected": int(is_sel),
                                 "is_high_freq": int(method in set(HIGH_FREQ))})
        n_pairs = n_app + n_not
        applicable = {m for m, v in rec["verdict_map"].items() if v == "APPLICABLE"}
        rows.append({
            "question_id": rec["question_id"], "task": rec["task"],
            "ssc": n_agree / n_pairs if n_pairs else None,
            "add_rate_in": n_add_in / n_not if n_not else None,
            "drop_rate": n_drop / n_app if n_app else None,
            "add_out_count": len(out_of_summary),
            "selection_accuracy": int(selected == gt),
            "applicable_set_size": len(applicable),
            "gt_recall": (len(applicable & gt) / len(gt) if gt else None),
        })

    with METRICS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if na_pairs:
        with NA_PAIR_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(na_pairs[0].keys()),
                               quoting=csv.QUOTE_ALL)
            w.writeheader()
            for r in na_pairs:
                w.writerow(r)

    _report(rows, pool)


def _pooled_ssc(pairs):
    if not pairs:
        return None, 0
    a = sum(1 for p in pairs
            if (p["verdict"] == "APPLICABLE" and p["selected"])
            or (p["verdict"] == "NOT_APPLICABLE" and not p["selected"]))
    return a / len(pairs), len(pairs)


def _pooled_add_in(pairs):
    nots = [p for p in pairs if p["verdict"] == "NOT_APPLICABLE"]
    if not nots:
        return None, 0
    return sum(1 for p in nots if p["selected"]) / len(nots), len(nots)


def _report(rows, pool):
    print("=" * 72)
    print(f"Exp 2 Cond C — Qwen2.5-72B (all {len(rows)} in main; oracle)")

    def avg(key):
        v = [r[key] for r in rows if r[key] is not None]
        return mean(v) if v else float("nan")
    print(f"  Mean SSC:            {avg('ssc'):.3f}")
    print(f"  Mean Add Rate(in):   {avg('add_rate_in'):.3f}")
    print(f"  Mean Drop Rate:      {avg('drop_rate'):.3f}")
    print(f"  Selection Accuracy:  {avg('selection_accuracy'):.3f}  "
          "(<1.0 strongly supports H1)")
    print(f"  Out-of-summary adds: {sum(r['add_out_count'] for r in rows)}")

    hi = [p for p in pool if p["method"] in set(HIGH_FREQ)]
    lo = [p for p in pool if p["method"] not in set(HIGH_FREQ)]
    print("\nPooled (sample,method) SSC / Add(in):")
    for label, pairs in (("High-freq", hi), ("Other", lo), ("ALL", pool)):
        ssc, n = _pooled_ssc(pairs)
        add, nna = _pooled_add_in(pairs)
        sscs = f"{ssc:.3f}" if ssc is not None else "n/a"
        adds = f"{add:.3f}" if add is not None else "n/a"
        print(f"  {label:12s} SSC={sscs} (n={n})  Add(in)={adds} (NA={nna})")

    print("\nHigh-freq Add Rate(in) per method:")
    for m in HIGH_FREQ:
        nots = [p for p in pool
                if p["method"] == m and p["verdict"] == "NOT_APPLICABLE"]
        if not nots:
            print(f"  {m:42s} no NA pairs")
            continue
        a = sum(1 for p in nots if p["selected"])
        flag = "" if len(nots) >= 5 else "  (low power)"
        print(f"  {m:42s} {a}/{len(nots)}={a/len(nots):.2f}{flag}")

    rob = [p for p in pool if p["method"] in set(ROBUST_OTHER)]
    present = {p["method"] for p in rob}
    ssc, n = _pooled_ssc(rob)
    sscs = f"{ssc:.3f}" if ssc is not None else "n/a"
    print(f"\nGeneralisation group: {len(present)}/{len(ROBUST_OTHER)} "
          f"methods appear, pooled SSC={sscs} (n={n})")
    absent = set(ROBUST_OTHER) - present
    if absent:
        print(f"  Not present (no S50 candidate set): {sorted(absent)}")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", choices=["oracle", "turn2", "metrics"],
                        default=None)
    args = parser.parse_args()

    if args.dry_run:
        construct_oracle()
        run_turn2(dry_run=True)
        return

    if args.stage in (None, "oracle"):
        print("--- Stage: construct oracle ---")
        construct_oracle()
    if args.stage in (None, "turn2"):
        print("--- Stage: Turn 2 ---")
        run_turn2()
    if args.stage in (None, "metrics"):
        print("--- Stage: metrics ---")
        compute_metrics()


if __name__ == "__main__":
    main()