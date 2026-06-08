# -*- coding: utf-8 -*-
import sys
import os
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder_path)

from vllm import LLM, SamplingParams
import pandas as pd
import time
import argparse 


model_ans_path = "Model Answer/"
prompt_dataset_path_test = "statqa/"


'''
Use Qwen to generate answers
- tricks: tricks involved, which will be remarked in output file name.
'''
def llama_answer_generation(model_type: str, dataset_name: str, output_name: str, trick: str):
    model_load_start_time = time.time()
    
    # Path settings
 
    if model_type == 'qwen2.5_7b':                                     
        model_path = "/rds/general/user/yx3422/home/qwen2.5-7b"
        parallel_num = 1
    elif model_type == 'qwen2.5_14b':                                       
        model_path = "/rds/general/user/yx3422/home/qwen2.5-14b"
        parallel_num = 1
    else:
        raise ValueError("[!] Invalid model type. Please choose from: 2_7b, 2_13b, 3_8b_instruct and 3_8b")
    
    # max token limit settings
    if trick == 'zero-shot-CoT' or trick == 'one-shot-CoT' or trick == 'plan-and-solve' or trick == 'PS_AE_check':
        max_tokens_limit = 2048
    else:
        max_tokens_limit = 256

    answer_store_path = "Origin Answer/"
    prompt_dataset_path = prompt_dataset_path_test
    file_path = prompt_dataset_path + dataset_name + ' for ' + trick + '.csv'
    output_path = model_ans_path + answer_store_path + output_name + model_type + '_' + trick + '.csv'
    answer_column_name = "model_answer"

    # Initialize model and sampling parameters
    llm = LLM(
        model=model_path,
        tensor_parallel_size=parallel_num,  # Adjust according to GPU configuration
        trust_remote_code=True
    )
    sampling_params = SamplingParams(
        temperature=0.0, 
        top_p=1.0, 
        max_tokens=max_tokens_limit
    )

    # Read dataset with prompt
    df = pd.read_csv(file_path)
    if answer_column_name not in df.columns:
        df[answer_column_name] = ''

    # Model loading end, generation start
    model_load_end_time = time.time()
    model_load_total_time = model_load_end_time - model_load_start_time
    generation_start_time = time.time()

    # Processing quantity per batch
    answer_batch_size = 5

    # Process prompts in batches and generate Llama's answers
    for i in range(0, len(df), answer_batch_size):
        batch_df = df.iloc[i:i+answer_batch_size]
    
        # Record valid indices and prompts
        valid_indices = []
        prompts = []
        for j, (idx, row) in enumerate(batch_df.iterrows()):
            if pd.isna(row[answer_column_name]) or row[answer_column_name] == '':
                valid_indices.append(idx)
                raw_prompt = row['prompt']
                if model_type in ['qwen2.5_7b', 'qwen2.5_14b']:    
                    formatted_prompt = (
                        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        f"<|im_start|>user\n{raw_prompt}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )                  
                    
                else:
                    formatted_prompt = raw_prompt
                prompts.append(formatted_prompt)    
    
        if not prompts:
            continue
        outputs = llm.generate(prompts, sampling_params)

        for k, out in enumerate(outputs[:2]):
            print(f"\n=== DEBUG OUTPUT {k} ===")
            print("TEXT:", repr(out.outputs[0].text))
            print("FINISH REASON:", out.outputs[0].finish_reason)

    
        sys.stdout.flush()
    
        # Write back using real indices
        for idx, output in zip(valid_indices, outputs):
            response = output.outputs[0].text
            df.at[idx, answer_column_name] = response

        # Update csv dataset
        df.to_csv(output_path, index=False)
        print(f"[+] (Current model: {model_type}, Prompting stategy: {trick}) Updated after processing batch starting from row {i}.")

    print(f"[i] csv file is fully updated with llama model-{model_type}-{trick} answers.")
    # Generation end
    generation_end_time = time.time()
    generation_total_time = generation_end_time - generation_start_time
    # Print information
    print("----------------------------------------------------------------------------------")
    print(f"[i] Model loading time comsumption: {model_load_total_time} seconds.")
    print(f"[i] Generation time comsumption: {generation_total_time} seconds.")
    print("----------------------------------------------------------------------------------")


# main
if __name__ == "__main__": 
    # Set the command line parameter parser
    parser = argparse.ArgumentParser(description='Generate answers using LLaMA.')
    parser.add_argument('--model_type', type=str, default='2_7b', help="model type. Please choose from: '2_7b', '2_13b', '3_8b_instruct','3_8b', 'qwen2.5_7b', 'qwen2.5_14b'.")
    parser.add_argument('--dataset_name', type=str, default='mini-StatQA', help='The name of the dataset file without extension')
    parser.add_argument('--output_name', type=str, default="llama", help='The base name for the output file')
    parser.add_argument('--trick', type=str, default='zero-shot', help="Prompting Strategy. Please choose from: 'zero-shot', 'one-shot', 'zero-shot-CoT', 'one-shot-CoT' and 'stats-prompt' (introducing domain knowledge).")
    
    # Parse command arguments
    args = parser.parse_args()
    
    # Call the function and pass in command arguments
    llama_answer_generation(
        model_type=args.model_type,
        dataset_name=args.dataset_name,
        output_name=args.output_name,
        trick=args.trick
    )