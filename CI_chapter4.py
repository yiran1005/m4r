# -*- coding: utf-8 -*-
"""
ci_from_statqa.py
=================
Extract per-task sample sizes (n) from the StatQA evaluation CSV and compute
Wilson score confidence intervals for accuracy (overall + per category) and for
error-type proportions.

WHY WILSON (not Wald / normal approximation):
    Every item is a Bernoulli trial (exact-match correct/incorrect, or
    belongs/does-not-belong to an error type). The Wald interval has poor
    coverage when p is near 0 or 1, or when n is small -- exactly our case
    (VT ~ 0, DS ~ 1, and 50-item annotation subsets). Wilson is robust there.

WHY A CONFIDENCE INTERVAL AT ALL:
    Decoding is deterministic (temperature=0, greedy), so model output has no
    randomness. The only source of uncertainty is the finite test set. The CI
    quantifies how the proportion would vary across resampled question sets.

WHAT THIS SCRIPT DOES:
    1. Reads the StatQA CSV and counts rows per value of the "task" column.
       That gives n per category (and the total n).
    2. Maps the raw task strings to the abbreviations used in the thesis tables
       (CA / CTT / DCT / VT / DS). Unmapped task names are reported so you can
       fix the mapping.
    3. Combines those n's with your measured accuracies (and error-type
       proportions) to print "point [lo, hi]" for every cell.

ASSUMPTION TO VERIFY:
    mini-StatQA is a fixed test set, so all conditions (zero-shot / CoT / plan,
    and all model sizes) are assumed to be evaluated on the SAME question set.
    Hence one CSV's per-task counts apply to the whole accuracy table. If some
    condition used a different subset, count n separately for that condition.

DEPENDENCIES:
    pip install pandas numpy scipy --break-system-packages
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, beta


# --------------------------------------------------------------------------- #
# Confidence-interval primitives                                              #
# --------------------------------------------------------------------------- #
def wilson_ci(x: int, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a proportion. Returns (p_hat, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = norm.ppf(1 - (1 - conf) / 2)            # 1.95996 for 95%
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return p, max(0.0, center - half), min(1.0, center + half)


def clopper_pearson_ci(x: int, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Clopper-Pearson exact interval (more conservative alternative)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    a = 1 - conf
    p = x / n
    lo = 0.0 if x == 0 else beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - a / 2, x + 1, n - x)
    return p, lo, hi


def ci_from_prop(p: float, n: int, conf: float = 0.95,
                 method: str = "wilson") -> tuple[float, float, float]:
    """
    CI when you only have the proportion p (not the success count):
    reconstruct x = round(p * n), then apply the chosen interval.
    Prefer passing exact counts to wilson_ci() when available, to avoid the
    small rounding error introduced here.
    """
    x = min(max(int(round(p * n)), 0), n)       # clamp to [0, n]
    return clopper_pearson_ci(x, n, conf) if method == "cp" else wilson_ci(x, n, conf)


# --------------------------------------------------------------------------- #
# Step 1 + 2: extract per-task n from the CSV                                 #
# --------------------------------------------------------------------------- #
# Edit this path to point at your CSV.
CSV_PATH = "/rds/general/user/yx3422/home/m4r_project/statqa/" \
           "mini-StatQA for plan-and-check-without-DK.csv"

TASK_COLUMN = "task"

# Map raw values in the "task" column -> thesis abbreviations.
# Adjust the keys to match EXACTLY what appears in your CSV (case / spelling).
TASK_MAP = {
    "Correlation Analysis":          "CA",
    "Contingency Table Test":        "CTT",
    "Distribution Compliance Test":  "DCT",
    "Variance Test":                 "VT",
    "Descriptive Statistics":        "DS",
}


def extract_n_by_category(csv_path: str = CSV_PATH,
                          task_col: str = TASK_COLUMN,
                          task_map: dict = TASK_MAP) -> dict[str, int]:
    """
    Read the CSV and return {"Overall": N, "CA": .., "CTT": .., ...}.
    Prints the raw value_counts and warns about any unmapped task names.
    """
    df = pd.read_csv(csv_path)
    if task_col not in df.columns:
        sys.exit(f"ERROR: column '{task_col}' not found. "
                 f"Available columns: {list(df.columns)}")

    raw_counts = df[task_col].value_counts(dropna=False)
    print("Raw value_counts of the 'task' column:")
    print(raw_counts.to_string())
    print(f"\nTotal rows (Overall n): {len(df)}\n")

    n_by_col: dict[str, int] = {"Overall": int(len(df))}
    unmapped = []
    for raw_value, count in raw_counts.items():
        abbr = task_map.get(str(raw_value).strip())
        if abbr is None:
            unmapped.append(raw_value)
        else:
            n_by_col[abbr] = n_by_col.get(abbr, 0) + int(count)

    if unmapped:
        print("WARNING: these task values were NOT mapped (fix TASK_MAP):")
        for u in unmapped:
            print(f"  - {u!r}")
        print()

    # Sanity check: per-category n should sum to Overall n.
    mapped_sum = sum(v for k, v in n_by_col.items() if k != "Overall")
    if mapped_sum != n_by_col["Overall"]:
        print(f"NOTE: mapped per-category n sums to {mapped_sum}, "
              f"but Overall n is {n_by_col['Overall']} "
              f"(difference = {n_by_col['Overall'] - mapped_sum}). "
              f"Check the mapping / unmapped values above.\n")

    print(f"Detected n_by_col = {n_by_col}\n")
    return n_by_col


# --------------------------------------------------------------------------- #
# Step 3: print accuracy / error tables with CIs                             #
# --------------------------------------------------------------------------- #
def report_table(rows: dict[str, list[float]],
                 col_order: list[str],
                 n_by_col: dict[str, int],
                 conf: float = 0.95,
                 method: str = "wilson") -> None:
    """
    Print "point [lo, hi]" for each condition x column.

    rows      : {condition_name: [proportions in col_order]}
    col_order : column labels, e.g. ["Overall","CA","CTT","DCT","VT","DS"]
    n_by_col  : {column_label: n}; columns without an n are skipped (printed as '-')
    """
    width = max((len(k) for k in rows), default=10) + 2
    header = f"{'condition':<{width}}" + "".join(f"{c:>22}" for c in col_order)
    print(header)
    print("-" * len(header))
    for cond, props in rows.items():
        line = f"{cond:<{width}}"
        for col, p in zip(col_order, props):
            n = n_by_col.get(col)
            if n is None or (isinstance(p, float) and np.isnan(p)):
                line += f"{'-':>22}"
                continue
            _, lo, hi = ci_from_prop(p, n, conf, method)
            line += f"{f'{p:.3f} [{lo:.3f},{hi:.3f}]':>22}"
        print(line)
    print(f"\n(method={method}, conf={conf:.0%})\n")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    # ---- get n from the CSV ----
    n_by_col = extract_n_by_category()

    # ===================================================================== #
    # ACCURACY TABLE (overall + per category). Replace with your measured   #
    # values if different. Columns must follow ACC_COLS order.              #
    # ===================================================================== #
    ACC_COLS = ["Overall", "CA", "CTT", "DCT", "VT", "DS"]
    accuracy_rows = {
        "7b-zero-shot":  [0.35512, 0.53571, 0.23047, 0.16525, 0.03279, 0.83784],
        "7b-zero-CoT":   [0.33362, 0.50595, 0.23438, 0.12712, 0.02049, 0.80309],
        "7b-plan":       [0.34566, 0.36310, 0.35547, 0.14831, 0.02869, 0.80309],
        "14b-zero-shot": [0.39725, 0.63690, 0.14844, 0.22034, 0.14344, 0.88803],
        "14b-zero-cot":  [0.46088, 0.55357, 0.26953, 0.30085, 0.29508, 0.89189],
        "14b-plan":      [0.47807, 0.52976, 0.38672, 0.28814, 0.29918, 0.87645],
        "72b-zero-shot": [0.47377, 0.50595, 0.44141, 0.44492, 0.03689, 0.92278],
        "72b-zero-CoT":  [0.46862, 0.53571, 0.44141, 0.43220, 0.00820, 0.91892],
        "72b-plan":      [0.38951, 0.20238, 0.27344, 0.24153, 0.50820, 0.64865],
    }
    print("=" * 90)
    print("ACCURACY with 95% Wilson CIs")
    print("=" * 90)
    report_table(accuracy_rows, ACC_COLS, n_by_col)

    # ===================================================================== #
    # ERROR-DISTRIBUTION TABLE. Each value is a proportion of the SAME total #
    # eval set, so they all use n = Overall. (4 main error types shown.)     #
    # Replace with your measured values.                                     #
    # ===================================================================== #
    ERR_COLS = ["IA", "CSE", "STC", "AE"]   # invalid / column / task-confusion / applicability
    error_rows = {
        "14b-zero-shot": [0.11608, 0.00774, 0.04729, 0.40241],
        "14b-zero-cot":  [0.04471, 0.01204, 0.10490, 0.34222],
        "14b-plan":      [0.00172, 0.01376, 0.04557, 0.43078],
        "72b-zero-shot": [0.00172, 0.01118, 0.08942, 0.39381],
        "72b-zero-CoT":  [0.00258, 0.01118, 0.08169, 0.40327],
        "72b-plan":      [0.25365, 0.00688, 0.03439, 0.30009],   # newly added 72B plan
    }
    # All error proportions use the overall n; build an n map keyed by ERR_COLS.
    err_n = {c: n_by_col["Overall"] for c in ERR_COLS}
    print("=" * 90)
    print("ERROR-TYPE PROPORTIONS with 95% Wilson CIs (n = Overall)")
    print("=" * 90)
    report_table(error_rows, ERR_COLS, err_n)

    # ---- optional: exact counts give slightly tighter/cleaner CIs ----
    # If you also have the per-cell success counts x, prefer:
    #   p, lo, hi = wilson_ci(x, n)
    # over ci_from_prop(p, n), to avoid the round(p*n) rounding step.