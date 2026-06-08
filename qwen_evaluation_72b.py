# -*- coding: utf-8 -*-
import os
import time
import argparse
import pandas as pd
from openai import OpenAI

model_ans_path = "Model Answer/"
prompt_dataset_path_test = "statqa/"

client = OpenAI(
    api_key="sk-ce07c64c3e1e48cd8b83a9ad25c0d0c3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def call_qwen_api(prompt, model_name, max_tokens, temperature=0.0, retries=5):
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=1.0,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"[!] API error at attempt {attempt + 1}: {e}")
            time.sleep(5 * (attempt + 1))
    return ""

def qwen_api_answer_generation(dataset_name: str, output_name: str, trick: str):
    model_name = "qwen2.5-72b-instruct"

    if trick in ["zero-shot-CoT", "one-shot-CoT", "PS_AE_check"]:
        max_tokens_limit = 4096
    else:
        max_tokens_limit = 256

    answer_store_path = "Origin Answer/"
    file_path = prompt_dataset_path_test + dataset_name + " for " + trick + ".csv"
    output_path = model_ans_path + answer_store_path + output_name + "_72b_" + trick + ".csv"
    answer_column_name = "model_answer"

    df = pd.read_csv(file_path)
    if answer_column_name not in df.columns:
        df[answer_column_name] = ""

    generation_start_time = time.time()

    for idx, row in df.iterrows():
        if pd.isna(row[answer_column_name]) or row[answer_column_name] == "":
            raw_prompt = row["prompt"]
            response = call_qwen_api(
                prompt=raw_prompt,
                model_name=model_name,
                max_tokens=max_tokens_limit,
                temperature=0.0
            )

            if idx < 2:
                print(f"\n=== DEBUG OUTPUT row {idx} ===")
                print(repr(response))

            if response:
                df.at[idx, answer_column_name] = response
                df.to_csv(output_path, index=False)
                print(f"[+] Updated row {idx}: {trick}")
                time.sleep(1)
            else:
                print(f"[!] Skipped row {idx}: all retries failed")

    generation_total_time = time.time() - generation_start_time
    print("-" * 82)
    print(f"[i] Finished Qwen2.5-72B generation for {trick}.")
    print(f"[i] Generation time consumption: {generation_total_time:.1f} seconds.")
    print("-" * 82)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers using Qwen API.")
    parser.add_argument("--dataset_name", type=str, default="mini-StatQA")
    parser.add_argument("--output_name", type=str, default="qwen2.5")
    parser.add_argument("--trick", type=str, default="zero-shot")
    args = parser.parse_args()

    qwen_api_answer_generation(
        dataset_name=args.dataset_name,
        output_name=args.output_name,
        trick=args.trick
    )