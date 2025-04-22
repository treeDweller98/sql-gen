#!/bin/bash

#SBATCH --partition=long
#SBATCH --time=1:00:00
#SBATCH -o log/grpo/log-%A-%a.out
#SBATCH -e log/grpo/log-%A-%a.err 
#SBATCH -J rl-llm
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=64G

module load cuda/12.6.0
source ~/scratch/ria-vlm-abs-rl/rl-llm/sql-gen/.venv/bin/activate
export WANDB_API_KEY=bd5a32650659a6513cd0c9da0fce7866fa7d3f2a
export RUN_NAME=$SLURM_JOB_NAME

models_list=(
    "qwen25_1_5b"
    "qwen25_3b"
    "qwen25_1_5b_instruct"
    "qwen25_3b_instruct"
    "qwen25_coder_1_5b_instruct"
    "qwen25_coder_3b_instruct"
    "granite_3b_code_instruct_2k"
    "granite_32_2b_instruct"
)
model="qwen25_coder_3b_instruct"

start_time=$(date +%s)
accelerate launch --config_file deepspeed.yaml --num_processes 1 \
    python main.py \
        --EXPERIMENT "grpo" \
        --MODEL $model \
        --OUTPUT_PATH "results/grpo/{model}_{experiment}/" \
        --INPUT_PATH "data/bird-train" \
        --BIRD_QUESTION_FILENAME "train.json" \
        --DB_FOLDERNAME "train_databases" \
        --DB_EXEC_TIMEOUT 30
end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
echo "Total time taken: $((elapsed_time / 3600)) hours $(( (elapsed_time % 3600) / 60 )) minutes $((elapsed_time % 60)) seconds"