#!/bin/bash

#SBATCH --array=0-5
#SBATCH --partition=long
#SBATCH --time=3:00:00
#SBATCH --output=log/slurm/log-%A-%a.log 
#SBATCH -J rl-llm
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=64G

conda activate /network/scratch/ria-vlm-abs-rl/rl-llm/sql-gen/sqlgen-env

export PATH=/network/scratch/i/islamria/ria-vllm-abs-rl/rl-llm/sql-gen/sqlgen-env/bin:$PATH
export PYTHONPATH=/network/scratch/i/islamria/ria-vllm-abs-rl/rl-llm/sql-gen:$PYTHONPATH

models_list=(
    "qwen25_coder_14b_instruct"
    "starcoder2_15b_instruct"
    "deepseek_coder_v2_lite_instruct"
    "granite_20b_code_instruct"
    "qwen25_coder_32b_instruct"
    "granite_34b_code_instruct"
)
model=${models_list[$SLURM_ARRAY_TASK_ID]}

python main.py \
    --EXPERIMENT "zs" \
    --MODEL $model \
    --GPU_MEMORY_UTILIZATION 0.97 \
    --TENSOR_PARALLEL_SIZE 4 \
    --MODEL_MAX_SEQ_LEN 4096 \
    --KV_CACHE_DTYPE "auto" \
    --VLLM_DTYPE "auto" \
    --SEED 42 \
    --BATCH_SIZE 128 \
    --INPUT_PATH "data/bird-minidev" \
    --OUTPUT_PATH "results/{model}_{experiment}/" \
    --BIRD_QUESTION_FILENAME "dev.json" \
    --DB_FOLDERNAME "dev_databases" \
    --DB_EXEC_TIMEOUT 30
    # --USE_CACHED_SCHEMA


# sbatch <file_Path>
# squeue -u islamria