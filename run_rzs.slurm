#!/bin/bash

#SBATCH --array=0-4
#SBATCH --partition=long
#SBATCH --time=4:00:00
#SBATCH --output=log/rzs_minidev-2/log-%A-%a.log 
#SBATCH -J rl-llm
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=64G

source activate /home/mila/i/islamria/scratch/ria-vlm-abs-rl/rl-llm/sql-gen/sqlgen-env

export PATH=/home/mila/i/islamria/scratch/ria-vllm-abs-rl/rl-llm/sql-gen/sqlgen-env/bin:$PATH
export PYTHONPATH=/home/mila/i/islamria/scratch/ria-vllm-abs-rl/rl-llm/sql-gen:$PYTHONPATH

models_list=(
    "deepseek_r1_qwen_1_5b"
    "deepseek_r1_qwen_7b"
    "deepseek_r1_qwen_14b"
    "deepseek_r1_qwen_32b"
    "qwq_32b"
)
model=${models_list[$SLURM_ARRAY_TASK_ID]}

start_time=$(date +%s)
python main.py \
    --EXPERIMENT "rzs" \
    --MODEL $model \
    --GPU_MEMORY_UTILIZATION 0.97 \
    --TENSOR_PARALLEL_SIZE 4 \
    --MODEL_MAX_SEQ_LEN 8192 \
    --KV_CACHE_DTYPE "auto" \
    --VLLM_DTYPE "auto" \
    --SEED 42 \
    --BATCH_SIZE 128 \
    --INPUT_PATH "data/bird-minidev" \
    --OUTPUT_PATH "results_new/rzs/{model}_{experiment}/" \
    --BIRD_QUESTION_FILENAME "dev.json" \
    --DB_FOLDERNAME "dev_databases" \
    --DB_EXEC_TIMEOUT 30
    # --USE_CACHED_SCHEMA
end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
echo "Total time taken: $((elapsed_time / 3600)) hours $(( (elapsed_time % 3600) / 60 )) minutes $((elapsed_time % 60)) seconds"


# sbatch <file_Path>
# squeue -u islamria