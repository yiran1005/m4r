"""
Sample 50 correct (selection_overall == 1) samples from the 14B zero-shot CoT outputs
for use as S_correct control group in Exp 1 Part 2 (Instance Judgement Probe).

Purpose:
    Provide an unbiased baseline of model judgement capability against S_50 (AE samples),
    to disentangle whether low judgement accuracy in S_50 reflects general weakness
    or sample-specific difficulty.

Reproducibility:
    Fixed random seed so the sample is stable across runs and reviewers can audit.
    Output includes both the full sampled rows and a lightweight ID-only file.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---- Configuration ----
INPUT_CSV = (
    "/rds/general/user/yx3422/home/m4r_project/"
    "Model Answer/Processed Answer/"
    "qwen_zero-shot-CoTqwen2.5_14b_zero-shot-CoT.csv"
)
OUTPUT_DIR = Path(
    "/rds/general/user/yx3422/home/m4r_project/Model Answer/Processed Answer"
)
SAMPLE_SIZE = 50
RANDOM_SEED = 42  # Fixed for reproducibility — do not change without re-running downstream
CORRECTNESS_COL = "selection_overall"


def load_data(path: str) -> pd.DataFrame:
    """Load CSV with helpful error reporting."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: Input file not found: {path}")
    df = pd.read_csv(p)
    print(f"Loaded {len(df)} rows from {p.name}")
    print(f"Columns: {list(df.columns)}")
    return df


def filter_correct(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Keep only rows where the model answered correctly."""
    if col not in df.columns:
        sys.exit(
            f"ERROR: Column '{col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # Coerce to numeric in case the column is stored as string/float
    correctness = pd.to_numeric(df[col], errors="coerce")
    correct_mask = correctness == 1
    correct_df = df[correct_mask].copy()

    n_total = len(df)
    n_correct = len(correct_df)
    n_invalid = correctness.isna().sum()

    print(f"Total rows:          {n_total}")
    print(f"Correct (==1):       {n_correct} ({n_correct / n_total:.1%})")
    print(f"Incorrect (==0):     {(correctness == 0).sum()}")
    if n_invalid > 0:
        print(f"Non-numeric values:  {n_invalid} (excluded)")

    return correct_df


def sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Randomly sample n rows with a fixed seed."""
    if len(df) < n:
        sys.exit(
            f"ERROR: Only {len(df)} correct samples available, "
            f"but {n} requested. Reduce SAMPLE_SIZE or check data."
        )
    sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"\nSampled {n} rows with random_state={seed}")
    return sampled


def save_outputs(sampled: pd.DataFrame, out_dir: Path) -> None:
    """Save full sampled rows and a lightweight ID-only file for traceability."""
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "S_correct_50_full.csv"
    sampled.to_csv(full_path, index=False)
    print(f"\nSaved full sample to:     {full_path}")

    # Lightweight ID file — adjust the ID column name if your CSV uses a different one
    id_candidates = ["id", "question_id", "qid", "index", "Unnamed: 0"]
    id_col = next((c for c in id_candidates if c in sampled.columns), None)
    if id_col is not None:
        id_path = out_dir / "S_correct_50_ids.csv"
        sampled[[id_col]].to_csv(id_path, index=False)
        print(f"Saved ID-only file to:    {id_path}  (column: {id_col})")
    else:
        print(
            "Note: No standard ID column found. "
            "Full file uses the original row order as implicit ID."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_CSV, help="Path to input CSV")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory to write sampled files",
    )
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE, help="Sample size")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    df = load_data(args.input)
    correct_df = filter_correct(df, CORRECTNESS_COL)
    sampled = sample(correct_df, n=args.n, seed=args.seed)
    save_outputs(sampled, Path(args.output_dir))

    # Quick sanity check on task category distribution if column exists
    cat_candidates = ["task_category", "task_type", "category"]
    cat_col = next((c for c in cat_candidates if c in sampled.columns), None)
    if cat_col is not None:
        print(f"\nTask category distribution in sample ({cat_col}):")
        print(sampled[cat_col].value_counts().to_string())


if __name__ == "__main__":
    main()