#!/bin/bash

#SBATCH --array=0-7
#SBATCH --partition=long
#SBATCH --time=1:30:00
#SBATCH --output=log/zs_minidev-2/log-%A-%a.log 
#SBATCH -J rl-llm
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=64G

source activate /home/mila/i/islamria/scratch/ria-vlm-abs-rl/rl-llm/sql-gen/sqlgen-env

export PATH=/home/mila/i/islamria/scratch/ria-vllm-abs-rl/rl-llm/sql-gen/sqlgen-env/bin:$PATH
export PYTHONPATH=/home/mila/i/islamria/scratch/ria-vllm-abs-rl/rl-llm/sql-gen:$PYTHONPATH

models_list=(
    "qwen25_14b_instruct"
    "qwen25_coder_14b_instruct"
    "starcoder2_15b_instruct"
    "deepseek_coder_v2_lite_instruct"
    "granite_20b_code_instruct"
    "qwen25_32b_instruct"
    "qwen25_coder_32b_instruct"
    "granite_34b_code_instruct"
)
model=${models_list[$SLURM_ARRAY_TASK_ID]}

start_time=$(date +%s)
python main.py \
    --EXPERIMENT "zs" \
    --MODEL $model \
    --GPU_MEMORY_UTILIZATION 0.97 \
    --TENSOR_PARALLEL_SIZE 4 \
    --MODEL_MAX_SEQ_LEN 8192 \
    --KV_CACHE_DTYPE "auto" \
    --VLLM_DTYPE "auto" \
    --SEED 42 \
    --BATCH_SIZE 128 \
    --INPUT_PATH "data/bird-minidev" \
    --OUTPUT_PATH "results_new/zs/{model}_{experiment}/" \
    --BIRD_QUESTION_FILENAME "dev.json" \
    --DB_FOLDERNAME "dev_databases" \
    --DB_EXEC_TIMEOUT 30
    # --USE_CACHED_SCHEMA
end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
echo "Total time taken: $((elapsed_time / 3600)) hours $(( (elapsed_time % 3600) / 60 )) minutes $((elapsed_time % 60)) seconds"


# sbatch <file_Path>
# squeue -u islamria