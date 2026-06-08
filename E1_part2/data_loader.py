"""
data_loader.py — Load S50 and extract (question, column_info, ground_truth).
============================================================================

The S50 CSV stores the model's *original* prompt as a single long string in
the `prompt` column. We need three pieces from each row:

  1. The statistical question text.
  2. The column metadata block (data type / num_of_rows / is_normality).
  3. The ground-truth applicable-method set, for J1 evaluation.

"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config


# Match the "### Column Information:" block up to "### Statistical Question:".
# The (?s) flag lets `.` cross newlines.
COLUMN_INFO_RE = re.compile(
    r"###\s*Column Information:\s*(.*?)\s*###\s*Statistical Question:",
    re.S,
)
# Match the statistical question itself (up to the next "###" header).
QUESTION_RE = re.compile(
    r"###\s*Statistical Question:\s*(.*?)\s*###",
    re.S,
)


@dataclass
class S50Sample:
    """A single S50 row distilled to what Part 2 actually needs."""
    question_id: int                  # row index in the CSV (stable across runs)
    dataset: str
    task: str                         # StatQA task category
    difficulty: str
    question: str                     # the natural-language question
    column_info: str                  # multi-line block, one column per line
    ground_truth_methods: list[str]   # list of applicable methods
    ground_truth_columns: list[str]


def _extract_question_and_columns(prompt_text: str) -> tuple[str, str]:
    """Pull `question` and `column_info` substrings from the original prompt.

    Raises ValueError if either section is missing — that means the upstream
    CSV format changed and the regex needs updating.
    """
    col_match = COLUMN_INFO_RE.search(prompt_text)
    q_match = QUESTION_RE.search(prompt_text)
    if not col_match:
        raise ValueError("Could not find '### Column Information:' block.")
    if not q_match:
        raise ValueError("Could not find '### Statistical Question:' block.")
    return q_match.group(1).strip(), col_match.group(1).strip()


def load_s50(csv_path: Path | str | None = None) -> list[S50Sample]:
    """Load all 50 rows and return them as S50Sample objects.

    The row index serves as a stable question_id. We deliberately do NOT use
    `refined_question` as the id because two questions could be word-identical
    on different datasets — row index is unambiguous.
    """
    csv_path = Path(csv_path) if csv_path else config.S50_CSV_PATH
    df = pd.read_csv(csv_path)

    samples = []
    for idx, row in df.iterrows():
        try:
            question, column_info = _extract_question_and_columns(row["prompt"])
        except ValueError as exc:
            raise ValueError(f"Row {idx}: {exc}") from exc

        gt_obj = json.loads(row["ground_truth"])
        samples.append(S50Sample(
            question_id=int(idx),
            dataset=str(row["dataset"]),
            task=str(row["task"]),
            difficulty=str(row["difficulty"]),
            question=question,
            column_info=column_info,
            ground_truth_methods=list(gt_obj.get("methods", [])),
            ground_truth_columns=list(gt_obj.get("columns", [])),
        ))
    return samples


def smoke_test() -> None:
    """Print a one-line summary per sample. Useful for sanity-checking on a
    new machine before running real inference."""
    samples = load_s50()
    print(f"Loaded {len(samples)} S50 samples.")
    print()
    for s in samples[:3]:
        print(f"[{s.question_id}] task={s.task}  difficulty={s.difficulty}")
        print(f"  Q: {s.question[:80]}{'...' if len(s.question) > 80 else ''}")
        n_cols = len([ln for ln in s.column_info.splitlines() if ln.strip()])
        print(f"  Columns: {n_cols} lines of metadata")
        print(f"  GT methods: {s.ground_truth_methods}")
        print()


if __name__ == "__main__":
    smoke_test()