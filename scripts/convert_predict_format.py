import pandas as pd
import json

if __name__ == '__main__':
    # 1️⃣ fine-tuned LLaMA2-7B 的预测结果（jsonl）
    jsonl_path = (
        "/rds/general/user/yx3422/home/m4r_project/"
        "saves/llama2-7b/lora/sft/generated_predictions.jsonl"
    )

    with open(jsonl_path, "r", encoding="utf-8") as f:
        predictions = [json.loads(line)["predict"] for line in f]

    # 2️⃣ 复用 base model 的 Origin Answer（保证 prompt / GT 对齐）
    base_csv_path = (
        "/rds/general/user/yx3422/home/m4r_project/"
        "Model Answer/Origin Answer/llama2_7b_zero-shot.csv"
    )
    df = pd.read_csv(base_csv_path)


    # 3️⃣ 用 fine-tuned 输出替换 model_answer
    df["model_answer"] = predictions

    # 4️⃣ 保存为新的 Origin Answer（给 analysis pipeline 用）
    output_csv_path = (
        "/rds/general/user/yx3422/home/m4r_project/"
        "Model Answer/Origin Answer/llama2_7b_sft_zero-shot.csv"
    )
    df.to_csv(output_csv_path, index=False)

    print(f"[+] Saved fine-tuned LLaMA2-7B predictions to:\n{output_csv_path}")
