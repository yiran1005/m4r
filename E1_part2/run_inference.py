"""
run_inference.py — Instance-level applicability judgement (Part 2 main task).
============================================================================

Loops over the 50 S50 questions x 27 methods = 1350 independent queries and
appends each result to a JSONL file. Resumable: rerunning skips records that
already exist for a given (question_id, method) pair.

Usage
-----
    python run_inference.py
    python run_inference.py --dry-run        # show 3 example prompts, no model
    python run_inference.py --limit-q 5      # smoke test on 5 questions only
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import config
from classification_list import CLASSIFICATION_LIST
from data_loader import load_s50
from prompts import build_judgement_messages

# torch / transformers imported lazily inside the loading and generation
# functions so that --dry-run works without a GPU environment.


# --------------------------------------------------------------------------- #
# I/O helpers (mirror Part 1's approach)
# --------------------------------------------------------------------------- #
def load_done_keys(path: Path) -> set[tuple[int, str]]:
    """Set of (question_id, method) pairs already written to disk."""
    if not path.exists():
        return set()
    done: set[tuple[int, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((int(rec["question_id"]), rec["method"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Model loading (lazy)
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer():
    import torch                                                       # noqa
    from transformers import AutoModelForCausalLM, AutoTokenizer       # noqa

    dtype = getattr(torch, config.TORCH_DTYPE)

    print(f"[{datetime.now():%H:%M:%S}] Loading tokenizer: {config.MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, trust_remote_code=True)

    print(f"[{datetime.now():%H:%M:%S}] Loading model "
          f"(dtype={config.TORCH_DTYPE}, device_map={config.DEVICE_MAP}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=dtype,
        device_map=config.DEVICE_MAP,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[{datetime.now():%H:%M:%S}] Model loaded.")
    return model, tokenizer


def generate_response(model, tokenizer, messages: list[dict], seed: int) -> str:
    import torch                          # noqa
    from transformers import set_seed     # noqa

    set_seed(seed)

    with torch.inference_mode():
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(model.device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=config.DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
        )
        if config.DO_SAMPLE:
            gen_kwargs.update(temperature=config.TEMPERATURE, top_p=config.TOP_P)

        output_ids = model.generate(input_ids, **gen_kwargs)
        new_tokens = output_ids[0, input_ids.shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print 3 example prompts, do not load the model.")
    parser.add_argument("--limit-q", type=int, default=None,
                        help="Limit to first N questions (for smoke testing).")
    parser.add_argument("--output", type=Path, default=config.RAW_JUDGEMENTS_PATH)
    args = parser.parse_args()

    samples = load_s50()
    if args.limit_q is not None:
        samples = samples[:args.limit_q]

    n_methods = len(CLASSIFICATION_LIST)
    total = len(samples) * n_methods
    print(f"Questions:     {len(samples)}")
    print(f"Methods:       {n_methods}")
    print(f"Total queries: {total}")
    print(f"Output:        {args.output}")

    if args.dry_run:
        for i, sample in enumerate(samples[:1]):
            for method in CLASSIFICATION_LIST[:3]:
                print("=" * 70)
                msgs = build_judgement_messages(
                    sample.question, sample.column_info, method,
                )
                print(f"[Q{sample.question_id}, M={method}]")
                print(msgs[-1]["content"][:600] + "...")
        print("=" * 70)
        print("Dry run complete. No model loaded.")
        return

    done = load_done_keys(args.output)
    if done:
        print(f"Resuming — {len(done)} (question, method) pairs already done.")

    model, tokenizer = load_model_and_tokenizer()

    t_start = time.time()
    n_done_this_session = 0

    for s_idx, sample in enumerate(samples, start=1):
        for m_idx, method in enumerate(CLASSIFICATION_LIST, start=1):
            if (sample.question_id, method) in done:
                continue

            messages = build_judgement_messages(
                sample.question, sample.column_info, method,
            )

            t0 = time.time()
            try:
                response = generate_response(
                    model, tokenizer, messages, seed=config.SEED,
                )
                error = None
            except Exception as e:                  # noqa: BLE001
                response = ""
                error = repr(e)
            elapsed = time.time() - t0

            record = {
                "question_id": sample.question_id,
                "method": method,
                "task": sample.task,
                "difficulty": sample.difficulty,
                "ground_truth_methods": sample.ground_truth_methods,
                "response": response,
                "error": error,
                "elapsed_sec": round(elapsed, 2),
                "model_id": config.MODEL_ID,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            append_jsonl(args.output, record)
            n_done_this_session += 1

            # Compact progress line — printing every line is too noisy at 1350
            # queries. Print every 10 + the first/last per question.
            if m_idx == 1 or m_idx == n_methods or m_idx % 10 == 0:
                done_total = len(done) + n_done_this_session
                pct = done_total / total * 100
                eta_sec = (time.time() - t_start) / max(n_done_this_session, 1) \
                          * (total - done_total)
                print(f"[{s_idx:2d}/{len(samples)}] Q{sample.question_id:3d} "
                      f"m={m_idx:2d}/{n_methods}  {method:42s} "
                      f"{elapsed:4.1f}s  [{done_total}/{total} = {pct:5.1f}%, "
                      f"ETA {eta_sec/60:.0f}m]")

    total_min = (time.time() - t_start) / 60
    print(f"\nDone in {total_min:.1f} min. "
          f"Wrote {n_done_this_session} new records to {args.output}")


if __name__ == "__main__":
    main()
