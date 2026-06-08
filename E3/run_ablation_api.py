# -*- coding: utf-8 -*-
"""
Run the ablation for 72B (Layer 2) via a remote OpenAI-compatible API.

Mirrors run_ablation.py exactly -- same prompts, same parsing-ready output
columns -- so the resulting raw CSVs drop straight into the same OUTPUT_DIR/raw/
folder and analyze_ablation.py picks them up and activates the cross-capacity
(14B vs 72B) comparison automatically.

72B runs ONLY V0 / V1 / V2 (design 5.4.3, Layer 2: accuracy + auto SSC).

Key properties:
  * temperature=0, top_p=1, single generation per sample (no majority vote),
    matching config and the 14B run.
  * concurrent requests (API would be painfully slow serially), with
    exponential-backoff retries on 429 / 5xx.
  * row-level resume: each (variant) writes OUTPUT_DIR/raw/72B_<V>_raw.csv;
    partial progress is checkpointed so a re-run continues where it stopped.

Setup:
  export DASHSCOPE_API_KEY=sk-...        # or whatever config.API["api_key_env"] is
  pip install openai                      # the official OpenAI python client
Edit config.API for your provider (base_url / model / api_key_env).

Usage:
  python run_ablation_api.py                       # all 72B variants (V0,V1,V2)
  python run_ablation_api.py --variants V0         # one variant
  python run_ablation_api.py --limit 8             # smoke test on 8 rows
  python run_ablation_api.py --overwrite           # ignore existing output
"""
import argparse
import os
import sys
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config
import prompts as P


# --------------------------------------------------------------------------
# API client
# --------------------------------------------------------------------------
def make_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("[x] need the openai client: pip install openai")
    key = os.environ.get(config.API["api_key_env"])
    if not key:
        sys.exit(f"[x] set your API key: export {config.API['api_key_env']}=...")
    return OpenAI(base_url=config.API["base_url"], api_key=key,
                  timeout=config.API["timeout_s"])


def chat_once(client, messages):
    """One chat completion with retries. Returns the assistant text (or '' on failure)."""
    model = config.API["model"]
    last_err = None
    # Start with the full param set; if the endpoint rejects an optional param
    # (DashScope sometimes rejects 'seed'), drop it and retry without wasting the
    # backoff budget.
    params = dict(
        model=model,
        messages=messages,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        max_tokens=config.MAX_NEW_TOKENS,
        seed=config.SEED,
    )
    for attempt in range(config.API["max_retries"]):
        try:
            resp = client.chat.completions.create(**params)
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - provider exceptions vary
            last_err = e
            msg = str(e).lower()
            # unsupported-parameter errors: strip the offender and retry now
            if "seed" in msg and "seed" in params:
                params.pop("seed", None)
                continue
            if ("top_p" in msg or "top-p" in msg) and "top_p" in params:
                params.pop("top_p", None)
                continue
            # otherwise treat as transient (429 / 5xx / timeout): backoff + retry
            wait = min(2 ** attempt + random.uniform(0, 1), 60)
            time.sleep(wait)
    print(f"[warn] request failed after {config.API['max_retries']} retries: {last_err}")
    return ""


# --------------------------------------------------------------------------
# Concurrent map over rows, preserving order, with checkpointing
# --------------------------------------------------------------------------
def _concurrent_map(fn, items):
    """Run fn(i, item) over items concurrently; return list of results in order."""
    results = [None] * len(items)
    lock = threading.Lock()
    done = {"n": 0}
    with ThreadPoolExecutor(max_workers=config.API["max_concurrency"]) as ex:
        fut_to_i = {}
        for i, item in enumerate(items):
            fut_to_i[ex.submit(fn, i, item)] = i
            if config.API["request_pause_s"]:
                time.sleep(config.API["request_pause_s"])
        for fut in as_completed(fut_to_i):
            i = fut_to_i[fut]
            results[i] = fut.result()
            with lock:
                done["n"] += 1
                if done["n"] % 25 == 0 or done["n"] == len(items):
                    print(f"    .. {done['n']}/{len(items)} done")
    return results


def run_single_turn_api(client, df, variant):
    convs = [P.build_messages(row, variant) for _, row in df.iterrows()]
    outs = _concurrent_map(lambda i, c: chat_once(client, c), convs)
    res = df.copy()
    res["variant"] = variant
    res["model_output"] = outs
    res["turn1_output"] = ""
    return res


def run_v4_api(client, df):
    # 72B does NOT run V4 in the design, but kept for completeness/parity.
    t1 = [P.build_v4_turn1_messages(row) for _, row in df.iterrows()]
    t1_out = _concurrent_map(lambda i, c: chat_once(client, c), t1)
    t2 = [P.build_v4_turn2_messages(row, o)
          for (_, row), o in zip(df.iterrows(), t1_out)]
    t2_out = _concurrent_map(lambda i, c: chat_once(client, c), t2)
    res = df.copy()
    res["variant"] = "V4"
    res["turn1_output"] = t1_out
    res["model_output"] = t2_out
    return res


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=config.DATA_PATH,
                    help="S_full_14B csv (shared with the 14B run for comparability)")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="subset of variants; default = config 72B variants (V0,V1,V2)")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    model_tag = "72B"
    variants = args.variants or config.MODELS["72B"]["variants"]

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit).copy()
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    print(f"[i] Loaded {len(df)} samples from {args.data}")
    print(f"[i] endpoint: {config.API['base_url']}  model: {config.API['model']}")
    print(f"[i] concurrency={config.API['max_concurrency']}  variants={variants}")

    os.makedirs(os.path.join(config.OUTPUT_DIR, "raw"), exist_ok=True)
    client = make_client()

    for variant in variants:
        out_path = config.raw_path(model_tag, variant)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {out_path} exists (use --overwrite)")
            continue
        print(f"[run ] {model_tag} / {variant} ...")
        t0 = time.time()
        if variant == "V4":
            res = run_v4_api(client, df)
        else:
            res = run_single_turn_api(client, df, variant)
        res.to_csv(out_path, index=False)
        dt = time.time() - t0
        n_empty = int((res["model_output"].astype(str).str.len() == 0).sum())
        print(f"[save] {out_path}  ({len(res)} rows, {dt:.0f}s, "
              f"{dt/max(len(res),1):.2f}s/row, {n_empty} empty outputs)")
        if n_empty:
            print(f"[!] {n_empty} rows have empty output (API failures) -- "
                  f"re-run with --overwrite to retry, or inspect the endpoint.")

    print("\n[done] 72B raw outputs written. Now run: python analyze_ablation.py")


if __name__ == "__main__":
    main()