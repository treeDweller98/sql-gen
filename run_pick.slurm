#!/bin/bash

#SBATCH --array=0
#SBATCH --partition=long
#SBATCH --time=8:00:00
#SBATCH --output=log/pick_minidev/log-%A-%a.log 
#SBATCH -J rl-llm
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=64G

source activate /home/mila/i/islamria/scratch/ria-vlm-abs-rl/rl-llm/sql-gen/sqlgen-env

export PATH=/home/mila/i/islamria/scratch/ria-vllm-abs-rl/rl-llm/sql-gen/sqlgen-env/bin:$PATH
export PYTHONPATH=/home/mila/i/islamria/scratch/ria-vllm-abs-rl/rl-llm/sql-gen:$PYTHONPATH

models_list=(
    "qwen25_14b_instruct"
    "qwen25_7b_instruct"
)
model=${models_list[$SLURM_ARRAY_TASK_ID]}

python main.py \
    --EXPERIMENT "pick" \
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

## "use this to run on lightning.ai:"
# python main.py \
#     --EXPERIMENT "pick" \
#     --MODEL $model \
#     --GPU_MEMORY_UTILIZATION 0.9 \
#     --TENSOR_PARALLEL_SIZE 1 \
#     --MODEL_MAX_SEQ_LEN 4096 \
#     --KV_CACHE_DTYPE "auto" \
#     --VLLM_DTYPE "bfloat16" \
#     --SEED 42 \
#     --BATCH_SIZE 8 \
#     --INPUT_PATH "" \
#     --OUTPUT_PATH "results/{model}_{experiment}/" \
#     --BIRD_QUESTION_FILENAME "minidev/MINIDEV/mini_dev_sqlite.json" \
#     --DB_FOLDERNAME "minidev/MINIDEV/dev_databases" \
#     --DB_EXEC_TIMEOUT 30
