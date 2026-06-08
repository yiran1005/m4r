"""
exp2_condA_72b.py — Exp 2 Cond A (Self-Summary) on Qwen2.5-72B via API.

Sample set: shared S50 (same as 14B), so 14B-vs-72B is comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import api_client
from classification_list import (
    CLASSIFICATION_LIST, TASK_CANDIDATES, candidates_for_task, normalise_task,
)
from data_loader import load_s50

# Config
PROJECT_ROOT = Path(__file__).resolve().parent
S50_CSV = PROJECT_ROOT / "qwen2_5_14b_zero-shot-CoT_50_with_AE_component.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
TURN1_RAW = OUTPUT_DIR / "condA_72b_turn1_raw.jsonl"
TURN1_PARSED = OUTPUT_DIR / "condA_72b_turn1_parsed.jsonl"
TURN2_RAW = OUTPUT_DIR / "condA_72b_turn2_raw.jsonl"
METRICS_CSV = OUTPUT_DIR / "condA_72b_metrics.csv"
TURN1_MAX_TOKENS = 4096
TURN2_MAX_TOKENS = 512

HIGH_FREQ = [
    "Anderson-Darling Test", "Fisher Exact Test", "Mantel-Haenszel Test",
    "Pearson Correlation Coefficient", "Bartlett Test", "F-Test for Variance",
]
CLASSIFICATION_LIST_STRING = ", ".join(CLASSIFICATION_LIST)
TASK_OPTIONS_STRING = "\n   ".join(f"- {t}" for t in TASK_CANDIDATES)


# Turn 1 prompt (closed-set Step 3, same as fixed 14B version)
TURN1_SYSTEM = (
    "You are a careful statistical assistant. You solve problems in stages. "
    "In this stage you only produce an applicability summary as a JSON object; "
    "you do NOT output a final method list yet."
)
TURN1_USER = """Let's solve this problem in stages. In this stage, you will only produce an applicability summary; you will NOT yet output the final answer.

Problem: {question}

Column information:
{column_info}

Classification list (for reference):
{classification_list}

Plan:
1. Identify the variables and their data types.
2. Determine the statistical task type. Choose EXACTLY ONE from this list, copying its name verbatim:
   {task_options}
3. From the classification list, identify ONLY the methods relevant to the task type determined in Step 2 (typically 3-8 methods).
4. For EACH of these candidate methods, evaluate whether it satisfies the data type, sample size, and statistical assumptions of this problem.

Output a JSON object with this exact structure:
{{
  "task_type": "<one of the five task types listed in Step 2, copied verbatim>",
  "summary": {{
    "<method name>": {{
      "verdict": "APPLICABLE" or "NOT_APPLICABLE",
      "reason": "<brief explanation>",
      "conditions_checked": ["<condition>: met/not_met", ...]
    }}
  }}
}}

Use method names exactly as written in the classification list, and the task_type exactly as written in Step 2. Output only the JSON object."""


# Turn 2 prompt (strict selection, fresh context)
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


# Stage 1: Turn 1
def build_turn1_messages(task: dict) -> list[dict]:
    return [
        {"role": "system", "content": TURN1_SYSTEM},
        {"role": "user", "content": TURN1_USER.format(
            question=task["question"], column_info=task["column_info"],
            classification_list=CLASSIFICATION_LIST_STRING,
            task_options=TASK_OPTIONS_STRING,
        )},
    ]


def turn1_record(task: dict, text: str, error: str | None) -> dict:
    return {
        "question_id": task["question_id"], "task": task["task"],
        "question": task["question"], "column_info": task["column_info"],
        "ground_truth_methods": task["ground_truth_methods"],
        "response": text, "error": error,
    }


def run_turn1(dry_run=False, limit_q=None):
    samples = load_s50(S50_CSV)
    if limit_q is not None:
        samples = samples[:limit_q]
    tasks = [{
        "question_id": s.question_id, "task": s.task, "question": s.question,
        "column_info": s.column_info,
        "ground_truth_methods": s.ground_truth_methods,
    } for s in samples]

    if dry_run:
        print(build_turn1_messages(tasks[0])[-1]["content"][:800])
        print("\n(dry run — no API calls)")
        return
    api_client.run_batch(tasks, build_turn1_messages, turn1_record,
                         TURN1_RAW, id_field="question_id",
                         max_tokens=TURN1_MAX_TOKENS)


# Stage 2: parse Turn 1
def extract_json(text: str):
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
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


def normalise_verdict(v):
    if not isinstance(v, str):
        return None
    k = v.strip().upper().replace(" ", "_").replace("-", "_")
    if k in ("APPLICABLE",):
        return "APPLICABLE"
    if k in ("NOT_APPLICABLE", "NOTAPPLICABLE", "INAPPLICABLE"):
        return "NOT_APPLICABLE"
    return None


def parse_turn1():
    n_ok = n_fail = n_step3_err = 0
    with TURN1_RAW.open() as fin, TURN1_PARSED.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            obj = extract_json(rec["response"])
            if obj is None:
                n_fail += 1
                continue
            gt_task = rec["task"]
            model_task = normalise_task(str(obj.get("task_type", "")))
            step3_correct = (model_task == gt_task)
            if not step3_correct:
                n_step3_err += 1

            raw_summary = obj.get("summary", obj)
            model_verdicts = {}
            for method, payload in raw_summary.items():
                if method not in CLASSIFICATION_LIST:
                    continue
                v = (normalise_verdict(payload.get("verdict", ""))
                     if isinstance(payload, dict)
                     else normalise_verdict(str(payload)))
                if v:
                    model_verdicts[method] = v

            candidates = candidates_for_task(gt_task)
            clean = {m: model_verdicts.get(m, "NOT_APPLICABLE") for m in candidates}
            if not clean:
                n_fail += 1
                continue
            n_ok += 1

            applicable = {m for m, v in clean.items() if v == "APPLICABLE"}
            gt = set(rec["ground_truth_methods"])
            recall = len(applicable & gt) / len(gt) if gt else None
            fout.write(json.dumps({
                "question_id": rec["question_id"], "task": gt_task,
                "question": rec["question"], "column_info": rec["column_info"],
                "ground_truth_methods": rec["ground_truth_methods"],
                "step3_correct": step3_correct,
                "verdict_map": clean,
                "applicable_set": sorted(applicable),
                "applicable_set_size": len(applicable),
                "gt_recall": recall,
            }, ensure_ascii=False) + "\n")

    print(f"Turn 1 parsed: {n_ok} OK, {n_fail} failed, "
          f"{n_step3_err} Step-3 errors (excluded from main analysis)")


# Stage 3: Turn 2 (fresh context)
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
        "step3_correct": task["step3_correct"], "verdict_map": task["verdict_map"],
        "ground_truth_methods": task["ground_truth_methods"],
        "response": text, "error": error,
    }


def run_turn2(dry_run=False):
    parsed = [json.loads(l) for l in TURN1_PARSED.open() if l.strip()]
    if dry_run:
        print(build_turn2_messages(parsed[0])[-1]["content"])
        print("\n(dry run — no API calls)")
        return
    api_client.run_batch(parsed, build_turn2_messages, turn2_record,
                         TURN2_RAW, id_field="question_id",
                         max_tokens=TURN2_MAX_TOKENS)


# Stage 4: metrics
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
    for rec in records:
        selected = parse_selection(rec["response"])
        gt = set(rec["ground_truth_methods"])
        candidates = set(candidates_for_task(rec["task"]))
        out_of_summary = selected - candidates
        n_app = n_not = n_agree = n_add_in = n_drop = 0
        for method, verdict in rec["verdict_map"].items():
            is_sel = method in selected
            pool.append({"qid": rec["question_id"], "method": method,
                         "verdict": verdict, "selected": is_sel,
                         "step3": rec["step3_correct"]})
            if verdict == "APPLICABLE":
                n_app += 1
                n_agree += int(is_sel); n_drop += int(not is_sel)
            else:
                n_not += 1
                n_agree += int(not is_sel); n_add_in += int(is_sel)
        n_pairs = n_app + n_not
        applicable = {m for m, v in rec["verdict_map"].items() if v == "APPLICABLE"}
        rows.append({
            "question_id": rec["question_id"], "task": rec["task"],
            "step3_correct": rec["step3_correct"],
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

    main = [r for r in rows if r["step3_correct"]]
    print("=" * 70)
    print(f"Exp 2 Cond A — Qwen2.5-72B")
    print(f"Total: {len(rows)}  main (step3 correct): {len(main)}")

    def avg(key, recs=main):
        v = [r[key] for r in recs if r[key] is not None]
        return mean(v) if v else float("nan")

    print(f"\nMain analysis (step3 correct):")
    print(f"  Mean SSC:            {avg('ssc'):.3f}")
    print(f"  Mean Add Rate(in):   {avg('add_rate_in'):.3f}")
    print(f"  Mean Drop Rate:      {avg('drop_rate'):.3f}")
    print(f"  Selection Accuracy:  {avg('selection_accuracy'):.3f}")
    print(f"  Mean APPLICABLE size:{avg('applicable_set_size'):.2f}")
    print(f"  Mean GT recall:      {avg('gt_recall'):.3f}")
    print("\n  (Section 5.2.4 reminder: read SSC together with APPLICABLE-set "
          "size and recall — narrow summary inflates SSC.)")

    main_qids = {r["question_id"] for r in main}
    pm = [p for p in pool if p["qid"] in main_qids]
    hi = [p for p in pm if p["method"] in set(HIGH_FREQ)]
    lo = [p for p in pm if p["method"] not in set(HIGH_FREQ)]

    def pooled_ssc(pairs):
        if not pairs:
            return None, 0
        a = sum(1 for p in pairs
                if (p["verdict"] == "APPLICABLE" and p["selected"])
                or (p["verdict"] == "NOT_APPLICABLE" and not p["selected"]))
        return a / len(pairs), len(pairs)
    print("\nPooled SSC (sample,method):")
    for label, pairs in (("High-freq", hi), ("Other", lo), ("ALL", pm)):
        ssc, n = pooled_ssc(pairs)
        print(f"  {label:12s} SSC={ssc:.3f} (n={n})" if ssc is not None
              else f"  {label:12s} n/a")
    print("=" * 70)


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", choices=["turn1", "parse", "turn2", "metrics"],
                        default=None, help="Run only one stage (default: all).")
    parser.add_argument("--limit-q", type=int, default=None)
    args = parser.parse_args()

    if args.dry_run:
        print("=== Turn 1 prompt ===")
        run_turn1(dry_run=True, limit_q=args.limit_q)
        return

    if args.stage in (None, "turn1"):
        print("--- Stage: Turn 1 ---")
        run_turn1(limit_q=args.limit_q)
    if args.stage in (None, "parse"):
        print("--- Stage: parse Turn 1 ---")
        parse_turn1()
    if args.stage in (None, "turn2"):
        print("--- Stage: Turn 2 ---")
        run_turn2()
    if args.stage in (None, "metrics"):
        print("--- Stage: metrics ---")
        compute_metrics()


if __name__ == "__main__":
    main()