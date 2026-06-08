"""
data_loader.py — Load S50 and extract (question, column_info, ground_truth).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COLUMN_INFO_RE = re.compile(
    r"###\s*Column Information:\s*(.*?)\s*###\s*Statistical Question:", re.S,
)
QUESTION_RE = re.compile(
    r"###\s*Statistical Question:\s*(.*?)\s*###", re.S,
)


@dataclass
class S50Sample:
    question_id: int
    dataset: str
    task: str
    difficulty: str
    question: str
    column_info: str
    ground_truth_methods: list[str]
    ground_truth_columns: list[str]


def _extract(prompt_text: str) -> tuple[str, str]:
    col = COLUMN_INFO_RE.search(prompt_text)
    q = QUESTION_RE.search(prompt_text)
    if not col:
        raise ValueError("missing '### Column Information:' block")
    if not q:
        raise ValueError("missing '### Statistical Question:' block")
    return q.group(1).strip(), col.group(1).strip()


def load_s50(csv_path):
    df = pd.read_csv(csv_path)
    samples = []
    for idx, row in df.iterrows():
        try:
            question, column_info = _extract(row["prompt"])
        except ValueError as exc:
            raise ValueError(f"Row {idx}: {exc}") from exc
        gt = json.loads(row["ground_truth"])
        samples.append(S50Sample(
            question_id=int(idx),
            dataset=str(row["dataset"]),
            task=str(row["task"]),
            difficulty=str(row["difficulty"]),
            question=question,
            column_info=column_info,
            ground_truth_methods=list(gt.get("methods", [])),
            ground_truth_columns=list(gt.get("columns", [])),
        ))
    return samples