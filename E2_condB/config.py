"""
Experiment 2, Condition C (Oracle Summary) — Configuration

Cond C has NO Turn 1. The verdict table is constructed deterministically from
StatQA ground truth:
  - methods in ground_truth     -> APPLICABLE
  - other candidates same task  -> NOT_APPLICABLE

Then the SAME Turn 2 as Cond A runs.
"""

import os
from pathlib import Path

# Model selection — Cond C runs on three models
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


# Decoding
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 42
TURN2_MAX_NEW_TOKENS = 512


# Data
PROJECT_ROOT = Path(__file__).resolve().parent
S50_CSV_PATH = PROJECT_ROOT / "qwen2.5_14b_zero-shot-CoT_50_with_AE_component.csv"

 
# I/O — outputs are namespaced by model so the three runs don't collide
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