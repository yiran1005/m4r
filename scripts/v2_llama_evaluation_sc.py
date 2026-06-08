# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
import time
import argparse
import json
import re
from collections import Counter

from vllm import LLM, SamplingParams

# paths
model_ans_path = "/rds/general/user/yx3422/home/m4r_project/Model Answer/"
prompt_dataset_path_test = "/rds/general/user/yx3422/home/m4r_project/statqa/"


# --------------------------------------------------
# Extract JSON from model output
# --------------------------------------------------
def extract_json(text):
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None
    return None


# --------------------------------------------------
# Majority vote using element-level frequency.
#
# Returns a tuple: (best_response, vote_method)
# vote_method is one of:
#   - 'majority_element'      : normal element-level majority vote
#   - 'fallback_exact_match'  : elements too scattered, fell back to exact-match vote
#   - 'fallback_parse_failure': all samples failed JSON parsing, returned responses[0]
# --------------------------------------------------
def majority_vote(responses):
    parsed = []

    for r in responses:
        result = extract_json(r)
        if result and 'columns' in result and 'methods' in result:
            parsed.append((result, r))

    if not parsed:
        # All samples failed JSON parsing – fall back to first response
        return responses[0], 'fallback_parse_failure'

    n_valid = len(parsed)
    threshold = n_valid / 2  # strict majority

    # Count element-level frequency
    col_counter = Counter()
    met_counter = Counter()
    for result, _ in parsed:
        col_counter.update(set(result['columns']))
        met_counter.update(set(result['methods']))

    # Keep elements that appear in more than half of valid samples
    majority_cols = {c for c, cnt in col_counter.items() if cnt > threshold}
    majority_mets = {m for m, cnt in met_counter.items() if cnt > threshold}

    # If nothing clears the threshold (very scattered), fall back to
    # the single most-frequent full-set response (original behaviour)
    if not majority_cols and not majority_mets:
        key_counter = Counter(
            (tuple(sorted(set(r['columns']))), tuple(sorted(set(r['methods']))))
            for r, _ in parsed
        )
        best_key = key_counter.most_common(1)[0][0]
        for result, r in parsed:
            if (tuple(sorted(set(result['columns']))),
                    tuple(sorted(set(result['methods'])))) == best_key:
                return r, 'fallback_exact_match'

    # Return the response whose answer best matches the majority sets
    # (maximise Jaccard overlap with majority_cols ∪ majority_mets)
    best_response = None
    best_score = -1
    for result, r in parsed:
        rc = set(result['columns'])
        rm = set(result['methods'])
        # Jaccard for columns + methods combined
        intersection = len(rc & majority_cols) + len(rm & majority_mets)
        union = len(rc | majority_cols) + len(rm | majority_mets)
        score = intersection / union if union > 0 else 0
        if score > best_score:
            best_score = score
            best_response = r

    return best_response, 'majority_element'


# --------------------------------------------------
# Main generation function
# --------------------------------------------------
def llama_answer_generation(model_type, dataset_name, output_name, trick, n_samples):

    model_load_start_time = time.time()

    # ------------------------
    # Model path
    # ------------------------
    if model_type == '2_7b':
        model_path = "/rds/general/user/yx3422/home/Llama-2-7b-chat-hf"
        parallel_num = 1

    elif model_type == '2_13b':
        model_path = "/rds/general/user/yx3422/home/Llama-2-13b-chat-hf"
        parallel_num = 1

    elif model_type == '3_8b_instruct':
        model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        parallel_num = 1

    elif model_type == '3_8b':
        model_path = "/rds/general/user/yx3422/home/Meta-Llama-3-8B"
        parallel_num = 1

    else:
        raise ValueError("Invalid model type.")

    # ------------------------
    # Token limit
    # ------------------------
    if trick in ('zero-shot-CoT', 'one-shot-CoT', 'self-consistency-CoT'):
        max_tokens_limit = 1024
    else:
        max_tokens_limit = 256

    # ------------------------
    # Sampling settings
    # ------------------------
    if trick == 'self-consistency-CoT':
        temperature = 0.5   # reduced from 0.7 to preserve JSON structure
        top_p = 0.9         # reduced from 0.95
    else:
        temperature = 0.0
        top_p = 1.0

    print(f"[i] Trick: {trick} | Temp: {temperature} | n_samples: {n_samples}")

    # ------------------------
    # File paths
    # ------------------------
    answer_store_path = "Origin Answer/"

    file_path = prompt_dataset_path_test + dataset_name + ' for ' + trick + '.csv'

    output_path = (
        model_ans_path
        + answer_store_path
        + output_name
        + model_type
        + '_'
        + trick
        + '.csv'
    )

    answer_column = "model_answer"

    # ------------------------
    # Load dataset
    # ------------------------
    df = pd.read_csv(file_path)

    if answer_column not in df.columns:
        df[answer_column] = ''

    if trick == 'self-consistency-CoT':
        if 'all_responses' not in df.columns:
            df['all_responses'] = ''
        if 'vote_method' not in df.columns:
            df['vote_method'] = ''

    # ------------------------
    # Load model
    # ------------------------
    llm = LLM(
        model=model_path,
        tensor_parallel_size=parallel_num,
        gpu_memory_utilization=0.85,
        max_num_seqs=4
    )

    # n=1, call multiple times for SC to avoid OOM
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens_limit,
        n=1,
        seed=None
    )

    model_load_end_time = time.time()
    print(f"[i] Model loaded in {model_load_end_time - model_load_start_time:.2f}s")

    # ------------------------
    # Generation
    # ------------------------
    generation_start = time.time()

    batch_size = 10

    for i in range(0, len(df), batch_size):

        batch_df = df.iloc[i:i + batch_size]
        prompts = batch_df['prompt'].tolist()

        valid_prompts = []
        valid_indices = []

        for j, prompt in enumerate(prompts):
            idx = i + j
            if pd.isna(df.iloc[idx][answer_column]) or df.iloc[idx][answer_column] == '':
                valid_prompts.append(prompt)
                valid_indices.append(idx)

        if not valid_prompts:
            continue

        for idx, prompt in zip(valid_indices, valid_prompts):

            if trick == 'self-consistency-CoT':

                # Call n_samples times with n=1 each to avoid OOM
                responses = []
                for _ in range(n_samples):
                    output = llm.generate([prompt], sampling_params)[0]
                    responses.append(output.outputs[0].text)

                response, vote_method = majority_vote(responses)
                df.at[idx, answer_column] = response
                df.at[idx, 'all_responses'] = json.dumps(responses, ensure_ascii=False)
                df.at[idx, 'vote_method'] = vote_method

            else:

                output = llm.generate([prompt], sampling_params)[0]
                response = output.outputs[0].text
                df.at[idx, answer_column] = response

        print(f"[+] Processed rows {i} - {i + batch_size}")

        # periodic save
        if i % 50 == 0:
            df.to_csv(output_path, index=False)

    df.to_csv(output_path, index=False)

    generation_end = time.time()
    print("--------------------------------------------------")
    print(f"[i] Generation time: {generation_end - generation_start:.2f}s")
    print("--------------------------------------------------")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type',    type=str, default='2_7b')
    parser.add_argument('--dataset_name',  type=str, default='mini-StatQA')
    parser.add_argument('--output_name',   type=str, default='llama')
    parser.add_argument('--trick',         type=str, default='zero-shot')
    parser.add_argument('--n_samples',     type=int, default=5)

    args = parser.parse_args()

    llama_answer_generation(
        model_type=args.model_type,
        dataset_name=args.dataset_name,
        output_name=args.output_name,
        trick=args.trick,
        n_samples=args.n_samples
    )