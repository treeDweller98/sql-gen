#!/bin/bash

#SBATCH --array=0-3
#SBATCH --partition=long
#SBATCH --time=8:00:00
#SBATCH --output=log/mad-minidev/log-%A-%a.log 
#SBATCH -J rl-llm
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=64G

source activate /home/mila/i/islamria/scratch/ria-vlm-abs-rl/rl-llm/sql-gen/sqlgen-env

export PATH=/home/mila/i/islamria/scratch/ria-vllm-abs-rl/rl-llm/sql-gen/sqlgen-env/bin:$PATH
export PYTHONPATH=/home/mila/i/islamria/scratch/ria-vllm-abs-rl/rl-llm/sql-gen:$PYTHONPATH

models_list=(
    "qwen25_14b_instruct"
    "qwen25_32b_instruct"
    "qwen25_coder_14b_instruct"
    "qwen25_coder_32b_instruct"
    # "granite_34b_code_instruct"
    # "deepseek_coder_v2_lite_instruct"
    # "deepseek_r1_qwen_14b"
    # "starcoder2_15b_instruct"
    # "granite_20b_code_instruct"
    # "deepseek_r1_qwen_32b"
)
model=${models_list[$SLURM_ARRAY_TASK_ID]}

python main.py \
    --EXPERIMENT "mad" \
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