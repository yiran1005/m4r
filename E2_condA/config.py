"""
Experiment 2, Condition A (Self-Summary) — Configuration
========================================================

Two-turn isolation, run as TWO separate scripts:
  Turn 1 (run_turn1.py): model generates an applicability summary (JSON).
  Turn 2 (run_turn2.py): fresh context, model selects from the clean verdict
                         table produced by post-processing Turn 1's output.

The two turns are deliberately separate processes that communicate only
through disk, so Turn 2 cannot leak any of Turn 1's reasoning context.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
MODEL_ID = "/rds/general/user/yx3422/home/qwen2.5-14b"
DEVICE_MAP = "auto"
TORCH_DTYPE = "bfloat16"

# --------------------------------------------------------------------------- #
# Decoding (matches Section 5.0.4)
# --------------------------------------------------------------------------- #
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 42

# Turn 1 needs room for a structured JSON summary of 3-8 methods. Section
# 5.2.2.1 specifies 4096 as a redundancy buffer against truncation.
TURN1_MAX_NEW_TOKENS = 4096
# Turn 2 only emits a short final selection JSON.
TURN2_MAX_NEW_TOKENS = 512

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent

# S50: the 50 14B zero-shot CoT AE samples (main analysis set).
S50_CSV_PATH = PROJECT_ROOT / "qwen2.5_14b_zero-shot-CoT_50_with_AE_component.csv"

# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Turn 1 raw model output (one record per question).
TURN1_RAW_PATH = OUTPUT_DIR / "turn1_raw.jsonl"

# Turn 1 parsed into structured summaries + the clean verdict table that
# feeds Turn 2. One record per question.
TURN1_PARSED_PATH = OUTPUT_DIR / "turn1_parsed.jsonl"

# Turn 2 raw model output (one record per question).
TURN2_RAW_PATH = OUTPUT_DIR / "turn2_raw.jsonl"

# Turn 2 parsed final selections.
TURN2_PARSED_PATH = OUTPUT_DIR / "turn2_parsed.jsonl"

# Final metrics CSV (SSC, Add/Drop rates, summary-size & recall, etc.)
METRICS_PATH = OUTPUT_DIR / "condA_metrics.csv"

# Parse-failure log (Turn 1 JSON errors or missing verdicts).
PARSE_FAILURE_LOG = OUTPUT_DIR / "turn1_parse_failures.jsonl"

# --------------------------------------------------------------------------- #
# Analysis subgroups
# --------------------------------------------------------------------------- #
# 6 high-frequency erroneous-retention methods (Section 5.2.5.1).
HIGH_FREQ_ERROR_METHODS = [
    "Anderson-Darling Test",
    "Fisher Exact Test",
    "Mantel-Haenszel Test",
    "Pearson Correlation Coefficient",
    "Bartlett Test",
    "F-Test for Variance",
]