#!/bin/bash
set -e

echo "======================================"
echo "Running LLaMA-2-7B-chat (ALL tricks)"
echo "======================================"

# 1. 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_eval

# 2. 单卡先跑通
export CUDA_VISIBLE_DEVICES=0

# 3. 进入脚本所在目录
cd "$(dirname "$0")"

# 4. 定义所有 prompting strategies
TRICKS=(
  "zero-shot"
  "one-shot"
  "zero-shot-CoT"
  "one-shot-CoT"
  "stats-prompt"
)

# 5. 依次运行每一种 strategy
for TRICK in "${TRICKS[@]}"; do
  echo "--------------------------------------"
  echo "Running strategy: $TRICK"
  echo "--------------------------------------"

  python llama_evaluation.py \
    --model_type 2_7b \
    --dataset_name mini-StatQA \
    --output_name llama \
    --trick "$TRICK"

  echo "Finished strategy: $TRICK"
done

echo "======================================"
echo "All strategies finished "
echo "======================================"

