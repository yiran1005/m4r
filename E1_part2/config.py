"""
Experiment 1, Part 2: Instance Judgement — Configuration
"""

from pathlib import Path


# Model (same as Part 1)
MODEL_ID = "/rds/general/user/yx3422/home/qwen2.5-14b"   
DEVICE_MAP = "auto"
TORCH_DTYPE = "bfloat16"


# Decoding (matches Section 5.0.4)
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_NEW_TOKENS = 512        # Smaller than Part 1: per-method judgement is short
SEED = 42


# Data
PROJECT_ROOT = Path(__file__).resolve().parent
S50_CSV_PATH = PROJECT_ROOT / "S_correct_50_full.csv"


# I/O — main experiment (Part 2 instance judgement)
OUTPUT_DIR = PROJECT_ROOT / "outputs_correct"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1350 records: 50 questions x 27 methods, each one query.
RAW_JUDGEMENTS_PATH = OUTPUT_DIR / "raw_judgements.jsonl"

# After parsing model output into {APPLICABLE, NOT_APPLICABLE, PARSE_ERROR}.
PARSED_JUDGEMENTS_PATH = OUTPUT_DIR / "parsed_judgements.jsonl"

# After comparing against ground truth.
J1_RESULTS_PATH = OUTPUT_DIR / "j1_results.csv"

# Stratified sample for human annotation of J2 / J3 / J4 (100 + 100 = 200).
ANNOTATION_CSV_PATH = OUTPUT_DIR / "for_annotation_j2_j3_j4.csv"

# I/O — C-Open leading-effect control
COPEN_OUTPUT_PATH = OUTPUT_DIR / "copen_responses.jsonl"
COPEN_PARSED_PATH = OUTPUT_DIR / "copen_parsed.jsonl"
COPEN_REPORT_PATH = OUTPUT_DIR / "copen_jaccard_report.csv"

# Number of questions used for the C-Open control
COPEN_NUM_QUESTIONS = 20

# Stratified sampling for J2-J4 annotation 
ANNOTATION_SAMPLE_PER_STRATUM = 100
ANNOTATION_SAMPLE_SEED = 2026