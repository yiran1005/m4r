# -*- coding: utf-8 -*-
"""
Configuration for the Plan-Components Ablation Study (Chapter 5.4).

Edit the paths / model registry here, or override most of them on the
command line (see run_ablation.py / analyze_ablation.py --help).
"""
import os

# Data
STATQA_RAW = "/rds/general/user/yx3422/home/m4r_project/statqa/mini-StatQA for zero-shot-CoT.csv"


DATA_PATH = "/rds/general/user/yx3422/home/m4r_project/statqa/S_full_14B.csv"

# Where raw model outputs and analysis results are written.
OUTPUT_DIR = "/rds/general/user/yx3422/home/m4r_project/E3/ablation_out"


MODELS = {
    "14B": {
        "model_path": "Qwen/Qwen2.5-14B-Instruct",
        "variants": ["V0", "V1", "V2", "V3", "V4"],
        "tensor_parallel_size": 1,      # single GPU
        # 14B-Instruct bf16 (~28GB weights) fits on one 48GB L40S, but leave
        # headroom for KV cache on long V2/V3 outputs -> 0.85, not 0.90.
        "gpu_memory_utilization": 0.85,
    },
    "72B": {
        "model_path": "Qwen/Qwen2.5-72B-Instruct",
        "variants": ["V0", "V1", "V2"],
        "tensor_parallel_size": 4,      # 72B needs more shards
        "gpu_memory_utilization": 0.92,
    },
}

ALL_VARIANTS = ["V0", "V1", "V2", "V3", "V4"]

# Variants that produce a Step-4 verdict block (used for auto SSC + label clarity).
# V4 has NO Step 4 (post-hoc self-check only) -> SSC is N/A for V4.
VARIANTS_WITH_STEP4 = ["V0", "V1", "V2", "V3"]

# Variants that are run as TWO turns (Turn-1 zero-shot CoT, Turn-2 self-check).
TWO_TURN_VARIANTS = ["V4"]

# Variants that get human label-clarity annotation sheets (50 samples each).
LABEL_CLARITY_VARIANTS = ["V0", "V1", "V2", "V3"]

# Decoding -- temperature 0, top_p 1, fixed seed.
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 42
MAX_NEW_TOKENS = 2048      # plan outputs are long; 2048 is a safe ceiling
MAX_MODEL_LEN = 8192       # context window cap for vLLM


# Statistics
BOOTSTRAP_ITERS = 10000
CI = 0.95
N_ANNOTATION_SAMPLES = 50  # samples per variant for human label-clarity sheet
ANNOTATION_SEED = 7


# API settings (for run_ablation_api.py -- the 72B layer-2 run via a remote
# OpenAI-compatible endpoint). 14B stays on local vLLM; only 72B uses the API.
API = {
    # Aliyun DashScope, OpenAI-compatible mode (official).
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    # DashScope model string for Qwen2.5-72B-Instruct:
    "model": "qwen2.5-72b-instruct",
    "api_key_env": "API_KEY",
    "max_concurrency": 8,      # parallel in-flight requests; lower if rate-limited
    "max_retries": 6,          # retries on 429 / 5xx with exponential backoff
    "timeout_s": 120,          # per-request timeout
    "request_pause_s": 0.0,    # optional fixed delay between submissions
}


def raw_path(model_tag: str, variant: str) -> str:
    """CSV of raw model outputs for one (model, variant)."""
    return os.path.join(OUTPUT_DIR, "raw", f"{model_tag}_{variant}_raw.csv")


def scored_path(model_tag: str, variant: str) -> str:
    """CSV of parsed + scored outputs for one (model, variant)."""
    return os.path.join(OUTPUT_DIR, "scored", f"{model_tag}_{variant}_scored.csv")


def annotation_path(model_tag: str, variant: str) -> str:
    """Human label-clarity annotation sheet for one (model, variant)."""
    return os.path.join(OUTPUT_DIR, "annotation", f"{model_tag}_{variant}_label_clarity_sheet.csv")