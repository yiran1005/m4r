# -*- coding: utf-8 -*-
"""
Build S_full_14B: the Qwen2.5-14B zero-shot-CoT samples that the model got
wrong with an APPLICABILITY ERROR (AE).

The source file does NOT carry a ready-made applicability_error column, so this
script COMPUTES the AE flag itself, using the *exact* logic of the StatQA repo's
Model Answer/Task Performance/error_type_analysis.py, so that the AE definition
here is identical to the one used for Chapter 4's error distribution.

AE definition (per row), reproducing the repo:
  1. NOT an invalid answer:
     at least one method from the 27-method classification list appears in
     `extracted_answer` (case-insensitive). If none appears -> Invalid Answer,
     never AE.
  2. methods comparison says: Correct > 0  AND  (Wrong + Missed) > 0.
     `methods_comparison_result` is a JSON string {"Correct":c,"Wrong":w,"Missed":m}.
     Correct>0  => at least one right method (the statistical task was understood)
     Wrong+Missed>0 => the method *selection* is imperfect (the AE signature).

Pure-AE vs AE-inclusive
-----------------------
The repo counts AE as a *single* error type: a row that ALSO has a column error
is bucketed as a MIXED error (CSE+AE), not pure AE. You can choose which set you
want for S_full_14B:

  --ae-mode pure      (default) the method-selection condition holds AND there is
                      NO column error (columns_score != 0). Matches the repo's
                      pure "Applicability Error (AE)" bucket and Chapter 4's
                      reported AE proportion.
  --ae-mode inclusive the method-selection condition holds regardless of column
                      error (i.e. pure AE + mixed CSE+AE). A larger set; use only
                      if your S_full_14B is meant to include mixed-error rows.

Source columns expected (from your file):
  dataset, refined_question, relevant_column, task, difficulty, results, prompt,
  model_answer, ground_truth, extracted_answer, methods_comparison_result,
  columns_comparison_result, methods_score, columns_score, selection_overall

Usage
-----
  python build_s_full.py --inspect-only        # report AE counts, write nothing
  python build_s_full.py                        # build pure-AE set -> config.DATA_PATH
  python build_s_full.py --ae-mode inclusive    # include mixed CSE+AE rows
  python build_s_full.py --src ... --out ...     # custom paths
"""
import argparse
import json
import os
import sys
import pandas as pd

import config
from statqa_core import ALL_METHODS

REQUIRED = ["dataset", "refined_question", "relevant_column",
            "task", "difficulty", "results", "prompt"]

# lower-cased method names for the invalid-answer check (repo: methods_list)
_METHODS_LOWER = [m.lower() for m in ALL_METHODS]


def _is_invalid_answer(extracted) -> bool:
    """Repo rule: invalid iff NO classification-list method appears in the text."""
    s = str(extracted).lower()
    return not any(m in s for m in _METHODS_LOWER)


def _methods_counts(cell):
    """Parse {"Correct":c,"Wrong":w,"Missed":m}; return (c, w+m) or None on failure."""
    try:
        d = json.loads(cell)
    except (TypeError, ValueError):
        return None
    c = d.get("Correct", 0)
    wm = d.get("Wrong", 0) + d.get("Missed", 0)
    return c, wm


def _has_column_error(row) -> bool:
    """Column error iff columns_score == 0 (repo uses columns_score==0)."""
    if "columns_score" in row and pd.notna(row["columns_score"]):
        try:
            return float(row["columns_score"]) == 0
        except (TypeError, ValueError):
            pass
    # fallback: parse columns_comparison_result if score missing
    cc = row.get("columns_comparison_result")
    parsed = _methods_counts(cc) if isinstance(cc, str) else None
    if parsed is not None:
        c, wm = parsed
        return not (c > 0 and wm == 0)
    return False


def compute_ae_flags(df, ae_mode):
    """Return a boolean Series: True where the row is an AE under the chosen mode."""
    flags = []
    n_invalid = 0
    n_method_cond = 0
    n_with_col_err = 0
    for _, row in df.iterrows():
        if _is_invalid_answer(row.get("extracted_answer", "")):
            n_invalid += 1
            flags.append(False)
            continue
        parsed = _methods_counts(row.get("methods_comparison_result"))
        if parsed is None:
            # unparseable methods result -> repo treats as task confusion, not AE
            flags.append(False)
            continue
        correct, wrong_missed = parsed
        method_cond = (correct > 0 and wrong_missed > 0)
        if method_cond:
            n_method_cond += 1
        col_err = _has_column_error(row)
        if col_err and method_cond:
            n_with_col_err += 1
        if ae_mode == "pure":
            flags.append(bool(method_cond and not col_err))
        else:  # inclusive
            flags.append(bool(method_cond))
    s = pd.Series(flags, index=df.index)
    print(f"[i] invalid answers (excluded): {n_invalid}")
    print(f"[i] rows meeting method AE condition (Correct>0 & Wrong+Missed>0): {n_method_cond}")
    print(f"[i]   of which also have a column error (mixed CSE+AE): {n_with_col_err}")
    print(f"[i] AE rows under --ae-mode {ae_mode}: {int(s.sum())}")
    return s


def inspect(src):
    df = pd.read_csv(src)
    print(f"[i] source: {src}")
    print(f"[i] columns ({len(df.columns)}): {list(df.columns)}")
    print(f"[i] total rows: {len(df)}")
    need = ["extracted_answer", "methods_comparison_result"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"[!] cannot compute AE -- missing columns: {missing}")
        return df
    print("\n[ pure AE ]")
    compute_ae_flags(df, "pure")
    print("\n[ inclusive AE (pure + mixed CSE+AE) ]")
    compute_ae_flags(df, "inclusive")
    miss_req = [c for c in REQUIRED if c not in df.columns]
    if miss_req:
        print(f"\n[!] required output columns missing (would need join): {miss_req}")
    else:
        print("\n[i] all 7 required output columns present.")
    return df


def build(src, out, ae_mode):
    df = pd.read_csv(src)
    for c in ("extracted_answer", "methods_comparison_result"):
        if c not in df.columns:
            sys.exit(f"[x] required column '{c}' not in source; cannot compute AE.")

    mask = compute_ae_flags(df, ae_mode)
    ae = df[mask].copy().reset_index(drop=True)
    if len(ae) == 0:
        sys.exit("[x] no AE rows after filtering -- check the source values.")

    miss_req = [c for c in REQUIRED if c not in ae.columns]
    if miss_req:
        sys.exit(f"[x] AE rows are missing required output columns {miss_req}. "
                 f"Your file has: {list(ae.columns)}")

    # keep the 7 required columns first, then carry useful eval columns for
    # downstream sanity-checks / stratification
    keep_extras = [c for c in ["model_answer", "ground_truth", "extracted_answer",
                               "methods_comparison_result", "columns_comparison_result",
                               "methods_score", "columns_score", "selection_overall"]
                   if c in ae.columns]
    ordered = REQUIRED + keep_extras
    ae = ae[ordered]
    ae.insert(0, "row_id", range(len(ae)))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ae.to_csv(out, index=False)
    print(f"[save] S_full_14B ({ae_mode} AE) -> {out}")
    print(f"[i] {len(ae)} rows, {len(ae.columns)} columns")
    if "task" in ae.columns:
        print("[i] task distribution:")
        print(ae["task"].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="/rds/general/user/yx3422/home/m4r_project/Model Answer/"
                "Processed Answer/qwen_zero-shot-CoTqwen2.5_14b_zero-shot-CoT.csv",
        help="processed-answer CSV with extracted_answer + methods_comparison_result")
    ap.add_argument(
        "--out", default=config.DATA_PATH,
        help="output path for S_full_14B (default: config.DATA_PATH, so "
             "run_ablation.py reads it directly)")
    ap.add_argument(
        "--ae-mode", choices=["pure", "inclusive"], default="pure",
        help="pure = AE without column error (repo's AE bucket, default); "
             "inclusive = AE regardless of column error (pure + mixed CSE+AE)")
    ap.add_argument("--inspect-only", action="store_true",
                    help="report AE counts under both modes, write nothing")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        sys.exit(f"[x] source not found: {args.src}\n    (pass --src)")

    if args.inspect_only:
        inspect(args.src)
        return

    inspect(args.src)
    print("-" * 60)
    build(args.src, args.out, args.ae_mode)


if __name__ == "__main__":
    main()