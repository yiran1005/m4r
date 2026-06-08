# -*- coding: utf-8 -*-
import pandas as pd
import os
import random
from collections import defaultdict

def stratified_sample_correct_and_wrong(
    input_csv: str,
    output_dir: str,
    total_correct: int = 50,
    total_wrong: int = 50,
    min_per_task: int = 8,
    random_seed: int = 42
):
    """
    Stratified sampling by task type.

    - Sample correct and wrong answers separately
    - Ensure each task has at least `min_per_task` samples if possible
    - Remaining quota is filled by random sampling across all tasks

    Conditions:
    - selection_overall == 1 → correct
    - selection_overall == 0 → wrong
    - extracted_answer != 'Invalid Answer'
    """

    random.seed(random_seed)

    df = pd.read_csv(input_csv)

    # Keep valid answers only
    df = df[df['extracted_answer'] != 'Invalid Answer']

    # Split correct / wrong
    df_correct = df[df['selection_overall'] == 1]
    df_wrong   = df[df['selection_overall'] == 0]

    tasks = df['task'].unique().tolist()

    def stratified_sample(df_pool, total_needed):
        sampled_rows = []
        remaining_pool = []

        # 1. Minimum per task
        for task in tasks:
            df_task = df_pool[df_pool['task'] == task]
            if len(df_task) == 0:
                continue

            n_sample = min(min_per_task, len(df_task))
            sampled = df_task.sample(n=n_sample, random_state=random_seed)
            sampled_rows.append(sampled)

            # Remaining for this task
            remaining_pool.append(df_task.drop(sampled.index))

        sampled_df = pd.concat(sampled_rows) if sampled_rows else pd.DataFrame()
        remaining_needed = total_needed - len(sampled_df)

        if remaining_needed < 0:
            # Too many sampled due to min_per_task, randomly downsample
            sampled_df = sampled_df.sample(n=total_needed, random_state=random_seed)
            return sampled_df

        # 2. Fill the rest randomly (ignoring task)
        if remaining_needed > 0:
            remaining_df = pd.concat(remaining_pool) if remaining_pool else pd.DataFrame()
            if len(remaining_df) < remaining_needed:
                raise ValueError("Not enough samples to fill remaining quota.")
            extra_sampled = remaining_df.sample(n=remaining_needed, random_state=random_seed)
            sampled_df = pd.concat([sampled_df, extra_sampled])

        return sampled_df

    # Perform stratified sampling
    sampled_correct = stratified_sample(df_correct, total_correct)
    sampled_wrong   = stratified_sample(df_wrong, total_wrong)

    # Output
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_csv))[0]

    correct_path = os.path.join(output_dir, f"{base_name}_correct_stratified_{total_correct}.csv")
    wrong_path   = os.path.join(output_dir, f"{base_name}_wrong_stratified_{total_wrong}.csv")

    sampled_correct.to_csv(correct_path, index=False)
    sampled_wrong.to_csv(wrong_path, index=False)

    print(f"[+] Stratified sampled correct examples → {correct_path}")
    print(f"[+] Stratified sampled wrong examples   → {wrong_path}")

    # Optional: print task distribution
    print("\n[Info] Task distribution (correct):")
    print(sampled_correct['task'].value_counts())

    print("\n[Info] Task distribution (wrong):")
    print(sampled_wrong['task'].value_counts())

    return correct_path, wrong_path


if __name__ == "__main__":
    stratified_sample_correct_and_wrong(
        input_csv="/rds/general/user/yx3422/home/m4r_project/Model Answer/Processed Answer/llama2_7b_zero-shot-CoT.csv",
        output_dir="/rds/general/user/yx3422/home/m4r_project/Model Answer/CoT_Analysis_Samples/",
        total_correct=50,
        total_wrong=50,
        min_per_task=8,
        random_seed=2024
    )

