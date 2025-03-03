#!/bin/bash
#SBATCH --array=0
#SBATCH --time=02:00:00   #02-00:00:00
#SBATCH --partition=long
#SBATCH --output=log/slurm/log-%A-%a.log 
#SBATCH -J rl-llm
#SBATCH --gres=gpu:rtx8000:2  #gpu:1
#SBATCH --mem=90G

conda activate /network/scratch/ria-vlm-abs-rl/rl-llm/sql-gen/sqlgen-env

export PATH=/network/scratch/i/islamria/ria-vllm-abs-rl/rl-llm/sql-gen/sqlgen-env/bin:$PATH
export PYTHONPATH=/network/scratch/i/islamria/ria-vllm-abs-rl/rl-llm/sql-gen:$PYTHONPATH

python main.py \
    --EXPERIMENT "zs" \
    --MODEL qwen25_coder_14b_instruct \
    --TENSOR_PARALLEL_SIZE 2 \
    --MODEL_MAX_SEQ_LEN 4096 \
    --KV_CACHE_DTYPE "auto" \
    --GPU_MEMORY_UTILIZATION 0.97 \
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