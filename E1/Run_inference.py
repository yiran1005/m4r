"""
run_inference.py — Query Qwen2.5-14B on each method in the classification list.
==============================================================================

For each of the 27 methods, prompt the model with the General Rule Recall
template and save the raw response. Output is one JSONL line per
(method, run_idx). The script is resumable: existing results are skipped.

"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import config
from classification_list import CLASSIFICATION_LIST
from prompts import build_messages

# torch and transformers are imported lazily inside load_model_and_tokenizer
# and generate_response so that --dry-run works without a CUDA environment.


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_done_keys(path: Path) -> set[tuple[str, int]]:
    """Return the set of (method, run_idx) pairs already present in path.

    Lets the script resume after an interruption without re-running queries.
    """
    if not path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["method"], rec["run_idx"]))
            except (json.JSONDecodeError, KeyError):
                # Skip malformed lines; they will be overwritten on next run
                # if you choose to clean the file manually.
                continue
    return done


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer():
    """Load Qwen2.5-14B-Instruct with the configured precision and placement."""
    import torch                                                       
    from transformers import AutoModelForCausalLM, AutoTokenizer      

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


# --------------------------------------------------------------------------- #
# Single-query generation
# --------------------------------------------------------------------------- #
def generate_response(model, tokenizer, messages: list[dict], seed: int) -> str:
    """Run one deterministic generation and return the assistant's reply text.

    We re-set the seed before every call so that, in theory, the same prompt
    always yields the same output regardless of call order. In practice
    transformers + flash attention can still introduce tiny non-determinism;
    this is acceptable for the experiment because Part 1 standardises on one
    output per method.
    """
    import torch                               
    from transformers import set_seed        

    set_seed(seed)

    with torch.inference_mode():
        # Use the model's own chat template (Qwen2.5 ships one).
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=config.DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
        )
        # Only attach sampling params if we ever decide to sample, to avoid
        # noisy warnings from transformers when do_sample=False.
        if config.DO_SAMPLE:
            gen_kwargs.update(temperature=config.TEMPERATURE, top_p=config.TOP_P)

        output_ids = model.generate(input_ids, **gen_kwargs)

        # Strip the prompt tokens; decode only the newly generated continuation.
        new_tokens = output_ids[0, input_ids.shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=config.NUM_RUNS_PER_METHOD,
                        help="Number of generations per method (default from config).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts only; do not load model or generate.")
    parser.add_argument("--output", type=Path, default=config.RAW_RESPONSES_PATH,
                        help="Output JSONL path.")
    args = parser.parse_args()

    methods = CLASSIFICATION_LIST
    print(f"Methods to query: {len(methods)}   Runs per method: {args.runs}")
    print(f"Total queries:    {len(methods) * args.runs}")
    print(f"Output file:      {args.output}")

    if args.dry_run:
        for method in methods[:3]:
            print("-" * 60)
            print(build_messages(method)[-1]["content"])
        print("-" * 60)
        print("Dry run complete. No model loaded.")
        return

    # Resume support: skip pairs we already have.
    done = load_done_keys(args.output)
    if done:
        print(f"Resuming — {len(done)} (method, run_idx) pairs already done.")

    model, tokenizer = load_model_and_tokenizer()

    t_start = time.time()
    for m_idx, method in enumerate(methods, start=1):
        for run_idx in range(args.runs):
            if (method, run_idx) in done:
                continue

            seed = config.SEED + run_idx  # 42, 43, 44, ...
            messages = build_messages(method)

            t0 = time.time()
            try:
                response = generate_response(model, tokenizer, messages, seed=seed)
                error = None
            except Exception as e:                  # noqa: BLE001
                response = ""
                error = repr(e)
            elapsed = time.time() - t0

            record = {
                "method": method,
                "run_idx": run_idx,
                "seed": seed,
                "prompt": messages[-1]["content"],
                "system_message": messages[0]["content"],
                "response": response,
                "error": error,
                "elapsed_sec": round(elapsed, 2),
                "model_id": config.MODEL_ID,
                "decoding": {
                    "do_sample": config.DO_SAMPLE,
                    "temperature": config.TEMPERATURE,
                    "top_p": config.TOP_P,
                    "max_new_tokens": config.MAX_NEW_TOKENS,
                },
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            append_jsonl(args.output, record)

            print(f"[{m_idx:2d}/{len(methods)}] run={run_idx} "
                  f"{method:40s} {elapsed:5.1f}s "
                  f"({len(response)} chars)"
                  + (f"  ERROR: {error}" if error else ""))

    total = time.time() - t_start
    print(f"\nDone in {total/60:.1f} min. Output: {args.output}")


if __name__ == "__main__":
    main()