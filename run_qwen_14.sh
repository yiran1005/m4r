#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -N Qwen-14b-all-tricks
#PBS -j oe   

set -e

echo "======================================"
echo "Running Qwen-14b all tricks"
echo "======================================"
echo "Start time: $(date)"

# activate env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_eval

echo "Python path:"
which python
python --version

# GPU info
echo "GPU status before running:"
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

cd /rds/general/user/yx3422/home/m4r_project

mkdir -p "Model Answer/Origin Answer/"

TRICKS=(
  "zero-shot"
  "zero-shot-CoT"
)

for TRICK in "${TRICKS[@]}"; do
  echo "--------------------------------------"
  echo "[INFO] Running strategy: $TRICK at $(date)"
  echo "--------------------------------------"

  python -u qwen_evaluation.py \
    --model_type qwen2.5_14b \
    --dataset_name mini-StatQA \
    --output_name "qwen_${TRICK}" \
    --trick "$TRICK" \
  || echo "[WARNING] $TRICK failed"

  echo "[INFO] Finished strategy: $TRICK"
done

echo "GPU status after running:"
nvidia-smi

echo "End time: $(date)"
echo "======================================"
echo "All strategies finished"
echo "======================================"