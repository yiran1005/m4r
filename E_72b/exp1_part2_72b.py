"""
exp1_part2_72b.py — Exp 1 Part 2 (Instance Judgement) on Qwen2.5-72B via API.
============================================================================

72B Layer-2 run: J1 automatic evaluation only, no human annotation
(Section 5.0.4 / 5.1.3). Runs all 50 questions x 27 methods = 1350 judgement
queries through DashScope, parses each verdict, and reports J1 accuracy
(overall, per task, per method) for cross-capacity comparison with 14B.

Sample set: the shared zero-shot-CoT S50 (same questions as 14B), so the
14B-vs-72B J1 comparison is on identical items.

Usage
-----
    export DASHSCOPE_API_KEY=sk-xxxx
    python api_client.py          # smoke-test the API first
    python exp1_part2_72b.py --dry-run
    python exp1_part2_72b.py            # full run (resumable)
    python exp1_part2_72b.py --eval-only   # re-run evaluation on existing output
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import api_client
from classification_list import CLASSIFICATION_LIST
from data_loader import load_s50

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
S50_CSV = PROJECT_ROOT / "qwen2_5_14b_zero-shot-CoT_50_with_AE_component.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
RAW_PATH = OUTPUT_DIR / "exp1_part2_72b_raw.jsonl"
J1_CSV = OUTPUT_DIR / "exp1_part2_72b_j1.csv"
MAX_TOKENS = 512

HIGH_FREQ = [
    "Anderson-Darling Test", "Fisher Exact Test", "Mantel-Haenszel Test",
    "Pearson Correlation Coefficient", "Bartlett Test", "F-Test for Variance",
]

# --------------------------------------------------------------------------- #
# Prompt (identical to the 14B Part 2 template)
# --------------------------------------------------------------------------- #
SYSTEM_MSG = (
    "You are a careful assistant that evaluates the applicability of "
    "statistical methods given concrete data features. Be precise. "
    "Always state APPLICABLE or NOT APPLICABLE in capital letters as the "
    "first line of your answer."
)
USER_TEMPLATE = """Problem: {question}

Column information:
{column_info}

Question: Is the method "{method_name}" applicable to this problem?
Please:
1. State your conclusion: APPLICABLE or NOT APPLICABLE.
2. Cite specific data features (e.g., data type, normality, sample size) as evidence.
3. For each known applicability condition of this method, explicitly state whether the data MEETS or DOES NOT MEET that condition."""


def build_messages(task: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_TEMPLATE.format(
            question=task["question"],
            column_info=task["column_info"],
            method_name=task["method"],
        )},
    ]


def make_record(task: dict, text: str, error: str | None) -> dict:
    return {
        "uid": task["uid"],
        "question_id": task["question_id"],
        "method": task["method"],
        "task": task["task"],
        "ground_truth_methods": task["ground_truth_methods"],
        "response": text,
        "error": error,
    }


# --------------------------------------------------------------------------- #
# Verdict parsing (same precedence ladder as 14B: NOT before APPLICABLE)
# --------------------------------------------------------------------------- #
NOT_RE = [re.compile(p, re.I) for p in [
    r"\bNOT\s+APPLICABLE\b", r"\bNOT\s+APPROPRIATE\b", r"\bis\s+not\s+applicable\b",
    r"\bcannot\s+be\s+(?:applied|used)\b", r"\bshould\s+not\s+be\s+(?:applied|used)\b",
    r"\bdoes\s+not\s+apply\b", r"\bINAPPLICABLE\b",
]]
APP_RE = [re.compile(p, re.I) for p in [
    r"\bAPPLICABLE\b", r"\bAPPROPRIATE\b", r"\bis\s+applicable\b",
    r"\bcan\s+be\s+(?:applied|used)\b", r"\bshould\s+be\s+(?:applied|used)\b",
]]


def parse_verdict(response: str) -> str:
    if not response or not response.strip():
        return "PARSE_ERROR"
    head = response[:200]
    for rx in NOT_RE:
        if rx.search(head):
            return "NOT_APPLICABLE"
    for rx in APP_RE:
        if rx.search(head):
            return "APPLICABLE"
    for rx in NOT_RE:
        if rx.search(response):
            return "NOT_APPLICABLE"
    for rx in APP_RE:
        if rx.search(response):
            return "APPLICABLE"
    return "PARSE_ERROR"


# --------------------------------------------------------------------------- #
# Build tasks
# --------------------------------------------------------------------------- #
def build_tasks() -> list[dict]:
    samples = load_s50(S50_CSV)
    tasks = []
    for s in samples:
        for m in CLASSIFICATION_LIST:
            tasks.append({
                "uid": f"{s.question_id}__{m}",
                "question_id": s.question_id,
                "method": m,
                "task": s.task,
                "question": s.question,
                "column_info": s.column_info,
                "ground_truth_methods": s.ground_truth_methods,
            })
    return tasks


# --------------------------------------------------------------------------- #
# J1 evaluation
# --------------------------------------------------------------------------- #
def evaluate() -> None:
    import csv
    records = []
    with RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    for r in records:
        verdict = parse_verdict(r["response"])
        expected = ("APPLICABLE" if r["method"] in set(r["ground_truth_methods"])
                    else "NOT_APPLICABLE")
        r["verdict"] = verdict
        r["expected"] = expected
        r["j1"] = ("parse_error" if verdict == "PARSE_ERROR"
                   else "correct" if verdict == expected else "wrong")

    with J1_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_id", "task", "method",
                                          "verdict", "expected", "j1"],
                           quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in
                        ["question_id", "task", "method", "verdict",
                         "expected", "j1"]})

    _report(records)


def _acc(recs):
    scored = [r for r in recs if r["j1"] in ("correct", "wrong")]
    if not scored:
        return 0.0, 0, 0
    c = sum(1 for r in scored if r["j1"] == "correct")
    return c / len(scored), c, len(scored)


def _report(records):
    n_total = len(records)
    n_err = sum(1 for r in records if r["j1"] == "parse_error")
    acc, c, n = _acc(records)
    print("=" * 70)
    print(f"Exp 1 Part 2 — Qwen2.5-72B (J1)")
    print(f"Total: {n_total}  parse errors: {n_err} "
          f"({n_err/n_total*100:.1f}%)  evaluable: {n}")
    print(f"Overall J1 accuracy: {acc*100:.1f}%  ({c}/{n})")
    print("\nPer-task:")
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    for t, recs in sorted(by_task.items()):
        a, cc, nn = _acc(recs)
        print(f"  {t:32s} {a*100:5.1f}%  ({cc}/{nn})")
    print("\nHigh-freq methods (compare with 14B's 90-98%):")
    by_method = defaultdict(list)
    for r in records:
        by_method[r["method"]].append(r)
    for m in HIGH_FREQ:
        a, cc, nn = _acc(by_method.get(m, []))
        print(f"  {m:42s} {a*100:5.1f}%  ({cc}/{nn})")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--limit-q", type=int, default=None)
    args = parser.parse_args()

    if args.eval_only:
        evaluate()
        return

    tasks = build_tasks()
    if args.limit_q is not None:
        keep_qids = set(range(args.limit_q))
        tasks = [t for t in tasks if t["question_id"] in keep_qids]

    print(f"Exp 1 Part 2 (72B): {len(tasks)} judgement queries")

    if args.dry_run:
        print("\nSample prompt:")
        print(build_messages(tasks[0])[-1]["content"][:600])
        print("\nDry run complete (no API calls).")
        return

    api_client.run_batch(
        tasks=tasks,
        build_messages=build_messages,
        make_record=make_record,
        output_path=RAW_PATH,
        id_field="uid",
        max_tokens=MAX_TOKENS,
    )
    print("\nEvaluating...")
    evaluate()


if __name__ == "__main__":
    main()