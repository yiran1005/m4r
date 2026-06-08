"""
Experiment 2, Condition C (Oracle Summary) — Configuration
==========================================================

Cond C has NO Turn 1. The verdict table is constructed deterministically from
StatQA ground truth (Section 5.2.4):
  - methods in ground_truth     -> APPLICABLE
  - other candidates same task  -> NOT_APPLICABLE

Then the SAME Turn 2 as Cond A runs (5.2.1.1 requires identical Turn 2 prompt).

Cond C runs on all three models (14B / 72B / 7B) — it is the strongest test of
H1 and the key evidence for a capacity threshold. Select the model via the
MODEL_KEY environment variable or by editing MODEL_KEY below.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Model selection — Cond C runs on three models (Section 5.2.7)
# --------------------------------------------------------------------------- #
MODELS = {
    "14b": "/rds/general/user/yx3422/home/qwen2.5-14b"
}

# Pick the model with: export MODEL_KEY=72b  (defaults to 14b).
MODEL_KEY = os.environ.get("MODEL_KEY", "14b").lower()
if MODEL_KEY not in MODELS:
    raise SystemExit(f"MODEL_KEY={MODEL_KEY!r} not in {list(MODELS)}")
MODEL_ID = MODELS[MODEL_KEY]

DEVICE_MAP = "auto"
TORCH_DTYPE = "bfloat16"

# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 42
TURN2_MAX_NEW_TOKENS = 512

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
S50_CSV_PATH = PROJECT_ROOT / "qwen2.5_14b_zero-shot-CoT_50_with_AE_component.csv"

# Optional: a CSV from 4.3.3 listing which (question, method) pairs the model
# actually erroneously retained in CoT. Used to mark the high-disconnect-risk
# subset of NA pairs (Section 5.2.4). If absent, that analysis is skipped and
# only the full-set Add Rate is reported. See construct_oracle.py docstring.
CoT_RETENTION_CSV = PROJECT_ROOT / "cot_retention_4_3_3.csv"
 

# --------------------------------------------------------------------------- #
# I/O — outputs are namespaced by model so the three runs don't collide
# --------------------------------------------------------------------------- #
OUTPUT_DIR = PROJECT_ROOT / f"outputs_{MODEL_KEY}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Oracle verdict tables (constructed, no model). Format matches Cond A's
# turn1_parsed.jsonl so run_turn2.py can consume it unchanged.
ORACLE_PARSED_PATH = OUTPUT_DIR / "oracle_parsed.jsonl"

# These two names mirror Cond A's config so the shared run_turn2.py works.
TURN1_PARSED_PATH = ORACLE_PARSED_PATH      # alias consumed by run_turn2.py
TURN2_RAW_PATH = OUTPUT_DIR / "turn2_raw.jsonl"

METRICS_PATH = OUTPUT_DIR / "condC_metrics.csv"
NA_PAIR_DETAIL_PATH = OUTPUT_DIR / "condC_na_pairs.csv"

# --------------------------------------------------------------------------- #
# Analysis subgroups
# --------------------------------------------------------------------------- #
HIGH_FREQ_ERROR_METHODS = [
    "Anderson-Darling Test",
    "Fisher Exact Test",
    "Mantel-Haenszel Test",
    "Pearson Correlation Coefficient",
    "Bartlett Test",
    "F-Test for Variance",
]

# The 12 judgement-robust "other" methods (J1 lower bound >= 85% across both
# populations) used as the generalisation-test group (Section 5.2.5.1).
# FILL THIS IN from your Exp 1 Part 2 S_correct comparison results. Left empty
# here; compute_metrics.py treats an empty list as "not yet specified" and
# skips the generalisation-group breakdown.
ROBUST_OTHER_METHODS: list[str] = [
    "Partial Correlation Coefficient",
    "Chi-square Independence Test",
    "Kolmogorov-Smirnov Test for Normality",
    "Kolmogorov-Smirnov Test for Uniform distribution",
    "Kolmogorov-Smirnov Test for Gamma distribution",
    "Kolmogorov-Smirnov Test for Exponential distribution",
    "Lilliefors Test",
    "Kendall Correlation Coefficient",
    "Levene Test",
    "Range",
    "Quartile",
    "Mode",
]