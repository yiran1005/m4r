# -*- coding: utf-8 -*-

import os
import time
import argparse
import pandas as pd
from openai import OpenAI

model_ans_path = "Model Answer/"
prompt_dataset_path_test = "statqa/"


client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=180.0,       
)


def call_qwen_api(prompt, model_name, max_tokens, temperature=0.0, retries=8):

    last_err = None
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            content = completion.choices[0].message.content
          
            if content is None or content.strip() == "":
                raise ValueError(f"Empty content (finish_reason="
                                 f"{completion.choices[0].finish_reason})")
            return content
        except Exception as e:
            last_err = e
            wait = 10 * (attempt + 1)
            print(f"[!] API error at attempt {attempt + 1}/{retries}: {e}. "
                  f"retrying in {wait}s")
            time.sleep(wait)
    print(f"[X] All {retries} retries failed. Last error: {last_err}")
    return ""


def qwen_api_answer_generation(dataset_name: str, output_name: str, trick: str):
    model_name = "qwen2.5-72b-instruct"

    # plan-and-solve 
    if trick in ["zero-shot-CoT", "one-shot-CoT", "PS_AE_check"]:
        max_tokens_limit = 4096
    else:
        max_tokens_limit = 256

    answer_store_path = "Origin Answer/"
    file_path = prompt_dataset_path_test + dataset_name + " for " + trick + ".csv"
    output_path = (model_ans_path + answer_store_path
                   + output_name + "_72b_" + trick + ".csv")
    answer_column_name = "model_answer"

   
    if os.path.exists(output_path):
        print(f"[i] Resuming from existing output: {output_path}")
        df = pd.read_csv(output_path)
        if answer_column_name not in df.columns:
            df[answer_column_name] = ""
    else:
        print(f"[i] Starting fresh from prompt file: {file_path}")
        df = pd.read_csv(file_path)
        if answer_column_name not in df.columns:
            df[answer_column_name] = ""

    # 统计待跑数
    todo_mask = df[answer_column_name].apply(
        lambda x: pd.isna(x) or str(x).strip() == ""
    )
    print(f"[i] Total rows: {len(df)}, rows to fill: {todo_mask.sum()}")

    if todo_mask.sum() == 0:
        print("[i] Nothing to do — all rows already filled.")
        return

    generation_start_time = time.time()
    filled, still_failed = 0, 0

    for idx, row in df.iterrows():
        if not (pd.isna(row[answer_column_name])
                or str(row[answer_column_name]).strip() == ""):
            continue

        response = call_qwen_api(
            prompt=row["prompt"],
            model_name=model_name,
            max_tokens=max_tokens_limit,
            temperature=0.0,
        )

        if idx < 2:
            print(f"\n=== DEBUG OUTPUT row {idx} ===")
            print(repr(response)[:400])

        if response:
            df.at[idx, answer_column_name] = response
            df.to_csv(output_path, index=False)
            filled += 1
            print(f"[+] Updated row {idx}: {trick}  ({filled} done)")
            time.sleep(1)
        else:
            still_failed += 1
            print(f"[!] Row {idx} still failed after all retries")

    elapsed = time.time() - generation_start_time
    print("-" * 82)
    print(f"[i] Finished Qwen2.5-72B generation for {trick}.")
    print(f"[i] Filled this run: {filled}, still failed: {still_failed}")
    print(f"[i] Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("-" * 82)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mini-StatQA")
    parser.add_argument("--output_name", type=str, default="qwen2.5")
    parser.add_argument("--trick", type=str, default="zero-shot")
    args = parser.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError(
            "DASHSCOPE_API_KEY not set. "
            "Run: export DASHSCOPE_API_KEY=sk-xxx  before launching."
        )

    qwen_api_answer_generation(
        dataset_name=args.dataset_name,
        output_name=args.output_name,
        trick=args.trick,
    )