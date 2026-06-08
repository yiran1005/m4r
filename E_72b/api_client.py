"""
api_client.py — Shared DashScope (Qwen) API client for the 72B experiments.
===========================================================================

All three 72B experiments (Exp 1 Part 2, Exp 2 Cond A, Exp 2 Cond C) call the
Qwen2.5-72B-Instruct model through Alibaba Cloud DashScope's OpenAI-compatible
endpoint. This module centralises:

  * client construction (base_url, api key from env)
  * a single chat() call with retry + exponential backoff on transient errors
  * concurrent batch execution with a bounded thread pool (rate-limit aware)
  * resumable JSONL output keyed by a caller-supplied id



Decoding matches the thesis protocol (Section 5.0.4): temperature=0, top_p=1.
DashScope accepts these via the OpenAI-compatible params.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from openai import OpenAI

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.environ.get("QWEN_MODEL", "qwen2.5-72b-instruct")

# Decoding (Section 5.0.4): deterministic.
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 42

# Concurrency + retry. DashScope rate limits vary by account tier; 5 concurrent
# requests is conservative. Lower MAX_WORKERS if you hit 429s frequently.
MAX_WORKERS = 5
MAX_RETRIES = 6
BACKOFF_BASE = 2.0          # seconds; doubles each retry (2, 4, 8, ...)
REQUEST_TIMEOUT = 120       # seconds per request


def get_client() -> OpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit(
            "DASHSCOPE_API_KEY not set. Run:  export DASHSCOPE_API_KEY=sk-xxxx"
        )
    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=REQUEST_TIMEOUT)


# --------------------------------------------------------------------------- #
# Single call with retry
# --------------------------------------------------------------------------- #
def chat(client: OpenAI, messages: list[dict],
         max_tokens: int) -> tuple[str, str | None]:
    """One chat completion. Returns (text, error). On success error is None.

    Retries on any exception with exponential backoff. After MAX_RETRIES the
    last error string is returned so the caller can record it and move on
    rather than crashing the whole batch.
    """
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=max_tokens,
                seed=SEED,
            )
            return resp.choices[0].message.content.strip(), None
        except Exception as e:                         # noqa: BLE001
            last_err = repr(e)
            if attempt < MAX_RETRIES - 1:
                sleep = BACKOFF_BASE * (2 ** attempt)
                time.sleep(sleep)
    return "", last_err


# --------------------------------------------------------------------------- #
# Resumable concurrent batch
# --------------------------------------------------------------------------- #
def load_done_ids(path: Path, id_field: str) -> set:
    """Return the set of ids already present in an output JSONL."""
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)[id_field])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_batch(
    tasks: list[dict],
    build_messages: Callable[[dict], list[dict]],
    make_record: Callable[[dict, str, str | None], dict],
    output_path: Path,
    id_field: str,
    max_tokens: int,
) -> None:
    """Run a batch of API calls concurrently, appending results to JSONL.

    Parameters
    ----------
    tasks         : list of task dicts (each must contain `id_field`)
    build_messages: task -> OpenAI messages list
    make_record   : (task, response_text, error) -> dict to write
    output_path   : JSONL output; appended to (resumable)
    id_field      : key identifying a task uniquely (for resume + dedupe)
    max_tokens    : max_tokens for each call

    Resumable: tasks whose id is already in output_path are skipped.
    Results are written as they complete (append mode), so an interrupted run
    loses nothing.
    """
    client = get_client()
    done = load_done_ids(output_path, id_field)
    pending = [t for t in tasks if t[id_field] not in done]
    print(f"Total tasks: {len(tasks)}  already done: {len(done)}  "
          f"to run: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    t_start = time.time()
    n_done = 0
    # A lock-free append is unsafe across threads, so we collect results in the
    # main thread as futures complete and write there.
    with output_path.open("a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            future_to_task = {}
            for task in pending:
                messages = build_messages(task)
                fut = pool.submit(chat, client, messages, max_tokens)
                future_to_task[fut] = task

            for fut in as_completed(future_to_task):
                task = future_to_task[fut]
                text, error = fut.result()
                record = make_record(task, text, error)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
                if n_done % 20 == 0 or n_done == len(pending):
                    rate = n_done / (time.time() - t_start)
                    eta = (len(pending) - n_done) / rate if rate else 0
                    print(f"  {n_done}/{len(pending)} done "
                          f"({rate:.1f}/s, ETA {eta/60:.1f}m)"
                          + ("  [errors present]" if error else ""))

    print(f"Batch done in {(time.time()-t_start)/60:.1f} min → {output_path}")


def smoke_test() -> None:
    """Verify the API key + endpoint work with one trivial call."""
    client = get_client()
    text, err = chat(client, [{"role": "user", "content": "Reply with OK."}],
                     max_tokens=10)
    if err:
        print(f"API smoke test FAILED: {err}")
    else:
        print(f"API smoke test OK. Model replied: {text!r}")
        print(f"Endpoint: {BASE_URL}")
        print(f"Model:    {MODEL}")


if __name__ == "__main__":
    smoke_test()