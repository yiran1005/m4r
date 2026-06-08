"""
run_turn1.py — Turn 1: Applicability Summary Generation (Cond A).
================================================================

One query per S50 question. The model determines the task type (Step 3) and
emits a JSON summary of per-method verdicts for the candidate set (Step 4).
We store the raw output; parsing/pruning happens in parse_turn1.py.

Resumable: skips question_ids already in the output JSONL.

Usage
-----
    python run_turn1.py
    python run_turn1.py --dry-run
    python run_turn1.py --limit-q 2
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import config
from data_loader import load_s50
from prompts import build_turn1_messages


def load_done_qids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["question_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-q", type=int, default=None)
    parser.add_argument("--output", type=Path, default=config.TURN1_RAW_PATH)
    args = parser.parse_args()

    samples = load_s50()
    if args.limit_q is not None:
        samples = samples[:args.limit_q]

    print(f"Turn 1 — questions: {len(samples)}")
    print(f"Output: {args.output}")

    if args.dry_run:
        s = samples[0]
        msgs = build_turn1_messages(s.question, s.column_info)
        print("=" * 70)
        print(msgs[-1]["content"][:1000] + "\n...")
        print("=" * 70)
        print("Dry run complete. No model loaded.")
        return

    done = load_done_qids(args.output)
    if done:
        print(f"Resuming — {len(done)} questions already done.")

    from inference_utils import load_model_and_tokenizer, generate
    model, tokenizer = load_model_and_tokenizer()

    t_start = time.time()
    n_new = 0
    for i, s in enumerate(samples, start=1):
        if s.question_id in done:
            continue
        messages = build_turn1_messages(s.question, s.column_info)
        t0 = time.time()
        try:
            response = generate(model, tokenizer, messages,
                                max_new_tokens=config.TURN1_MAX_NEW_TOKENS,
                                seed=config.SEED)
            error = None
        except Exception as e:               # noqa: BLE001
            response, error = "", repr(e)
        elapsed = time.time() - t0

        append_jsonl(args.output, {
            "question_id": s.question_id,
            "task": s.task,
            "difficulty": s.difficulty,
            "question": s.question,
            "column_info": s.column_info,
            "ground_truth_methods": s.ground_truth_methods,
            "response": response,
            "error": error,
            "elapsed_sec": round(elapsed, 2),
            "model_id": config.MODEL_ID,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        n_new += 1
        print(f"[{i:2d}/{len(samples)}] Q{s.question_id:3d} {s.task:30s} "
              f"{elapsed:5.1f}s ({len(response)} chars)"
              + (f"  ERROR: {error}" if error else ""))

    print(f"\nTurn 1 done in {(time.time()-t_start)/60:.1f} min. "
          f"Wrote {n_new} new records to {args.output}")


if __name__ == "__main__":
    main()