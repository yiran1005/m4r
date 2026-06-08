"""
run_turn2.py — Turn 2: Selection Conditioned on Summary (Cond A).
================================================================

Reads turn1_parsed.jsonl (clean verdict tables) and, in a FRESH context with
no Turn 1 history, asks the model to select only the APPLICABLE methods.

This is a separate process from Turn 1 — the strongest form of the context
reset in Section 5.2.1.1. The model never sees Turn 1's reasoning, only the
distilled verdict table.

By default, Step-3-incorrect questions are still run (so you have the data),
but they are flagged in the output for exclusion from the main analysis.

Resumable.

Usage
-----
    python run_turn2.py
    python run_turn2.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import config
from prompts import build_turn2_messages


def load_parsed(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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
    parser.add_argument("--input", type=Path, default=config.TURN1_PARSED_PATH)
    parser.add_argument("--output", type=Path, default=config.TURN2_RAW_PATH)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Turn 1 parsed not found: {args.input}. "
                         "Run parse_turn1.py first.")

    parsed = load_parsed(args.input)
    print(f"Turn 2 — questions: {len(parsed)}")
    print(f"Output: {args.output}")

    if args.dry_run:
        r = parsed[0]
        msgs = build_turn2_messages(r["question"], r["column_info"],
                                    r["verdict_map"])
        print("=" * 70)
        print(msgs[-1]["content"])
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
    for i, r in enumerate(parsed, start=1):
        qid = r["question_id"]
        if qid in done:
            continue
        messages = build_turn2_messages(r["question"], r["column_info"],
                                        r["verdict_map"])
        t0 = time.time()
        try:
            response = generate(model, tokenizer, messages,
                                max_new_tokens=config.TURN2_MAX_NEW_TOKENS,
                                seed=config.SEED)
            error = None
        except Exception as e:               # noqa: BLE001
            response, error = "", repr(e)
        elapsed = time.time() - t0

        append_jsonl(args.output, {
            "question_id": qid,
            "task": r["task"],
            "step3_correct": r["step3_correct"],
            "verdict_map": r["verdict_map"],
            "applicable_set": r["applicable_set"],
            "ground_truth_methods": r["ground_truth_methods"],
            "response": response,
            "error": error,
            "elapsed_sec": round(elapsed, 2),
            "model_id": config.MODEL_ID,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        n_new += 1
        print(f"[{i:2d}/{len(parsed)}] Q{qid:3d} {elapsed:4.1f}s "
              f"({len(response)} chars)"
              + (f"  ERROR: {error}" if error else ""))

    print(f"\nTurn 2 done in {(time.time()-t_start)/60:.1f} min. "
          f"Wrote {n_new} new records to {args.output}")


if __name__ == "__main__":
    main()