#!/bin/bash
#SBATCH --array=0-9         # Creates 10 runs (indices 0 to 9)
#SBATCH --time=10:00:00   #02-00:00:00
#SBATCH --partition=long
#SBATCH --output=log/slurm/log-%A-%a.log 
#SBATCH -J rl-llm
#SBATCH --gres=gpu:l40s:2  #gpu:1
#SBATCH --mem=60G

conda activate /network/scratch/ria-vlm-abs-rl/rl-llm/sql-gen/sqlgen-env

export PATH=/network/scratch/i/islamria/ria-vllm-abs-rl/rl-llm/sql-gen/sqlgen-env/bin:$PATH
export PYTHONPATH=/network/scratch/i/islamria/ria-vllm-abs-rl/rl-llm/sql-gen:$PYTHONPATH

EXPERIMENT_ALL=('1' '2' ....)

# Select seed based on SLURM array index
EXPERIMENT=${EXPERIMENT_ALL[$SLURM_ARRAY_TASK_ID]}

python main.py \
    --EXPERIMENT $EXPERIMENT \
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