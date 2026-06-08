"""
Experiment 1, Part 1: General Rule Recall — Configuration
========================================================

Centralized configuration.
"""

from pathlib import Path


# Model

MODEL_ID = "/rds/general/user/yx3422/home/qwen2.5-14b"

# Device map for transformers. 
DEVICE_MAP = "auto"

TORCH_DTYPE = "bfloat16"

# Decoding 
# Deterministic decoding: temperature=0, top_p=1, fixed seed.
# We run each (method) query ONCE because at temp=0 with a fixed seed the
# output is deterministic up to backend non-determinism. 
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_NEW_TOKENS = 768           
SEED = 42
NUM_RUNS_PER_METHOD = 1        

# I/O
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Raw model outputs, one JSON object per (method, run_idx).
RAW_RESPONSES_PATH = OUTPUT_DIR / "raw_responses.jsonl"

# After (optional) consistency picking, one row per method.
CONSISTENCY_PICKED_PATH = OUTPUT_DIR / "consistency_picked.jsonl"

# Annotator-facing CSV (one row per method, columns for D1–D5).
ANNOTATION_CSV_PATH = OUTPUT_DIR / "for_annotation.csv"

# Final scores computed after annotation is filled in.
SCORES_PATH = OUTPUT_DIR / "knowledge_scores.csv"

