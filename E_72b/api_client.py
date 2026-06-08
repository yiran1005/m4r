from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from openai import OpenAI

# Configuration
BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.environ.get("QWEN_MODEL", "qwen2.5-72b-instruct")

# Decoding : deterministic.
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 42

MAX_WORKERS = 5
MAX_RETRIES = 6
BACKOFF_BASE = 2.0          
REQUEST_TIMEOUT = 120      


def get_client() -> OpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit(
            "DASHSCOPE_API_KEY not set. Run:  export DASHSCOPE_API_KEY=sk-xxxx"
        )
    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=REQUEST_TIMEOUT)

# Single call with retry
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

# Resumable concurrent batch
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

    tasks whose id is already in output_path are skipped.
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