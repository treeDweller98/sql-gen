#!/bin/bash

#SBATCH --array=0-11
#SBATCH --partition=long
#SBATCH --time=1:30:00
#SBATCH -o log/zs_minidev/log-%A-%a.out
#SBATCH -e log/zs_minidev/log-%A-%a.err 
#SBATCH -J rl-llm
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=64G

source ~/scratch/ria-vlm-abs-rl/rl-llm/sql-gen/.venv/bin/activate

models_list=(
    "qwen25_7b_instruct"
    "qwen25_14b_instruct"
    "qwen25_32b_instruct"
    "qwen25_coder_7b_instruct"
    "qwen25_coder_14b_instruct"
    "qwen25_coder_32b_instruct"
    "granite_32_8b_instruct"
    "granite_3b_code_instruct_128k"
    "granite_8b_code_instruct_128k"
    "granite_20b_code_instruct_8k"
    "granite_34b_code_instruct_8k"
    "deepseek_v2_lite_chat"
    "deepseek_coder_v2_lite_instruct"
    # "deepseek_r1_qwen_7b"
    # "deepseek_r1_qwen_14b"
    # "deepseek_r1_qwen_32b"
    # "qwq_32b"
)
model=${models_list[$SLURM_ARRAY_TASK_ID]}

start_time=$(date +%s)
python main.py \
    --EXPERIMENT "zs" \
    --MODEL $model \
    --GPU_MEMORY_UTILIZATION 0.97 \
    --TENSOR_PARALLEL_SIZE 4 \
    --MODEL_MAX_SEQ_LEN 8192 \
    --DISTRIBUTED_EXECUTOR_BACKEND 'mp' \
    --KV_CACHE_DTYPE "auto" \
    --VLLM_DTYPE "auto" \
    --ENFORCE_EAGER \
    --ENABLE_PREFIX_CACHING \
    --SEED 42 \
    --BATCH_SIZE 128 \
    --INPUT_PATH "data/bird-minidev" \
    --OUTPUT_PATH "results/zs/{model}_{experiment}/" \
    --BIRD_QUESTION_FILENAME "dev.json" \
    --DB_FOLDERNAME "dev_databases" \
    --DB_EXEC_TIMEOUT 30 \
    # --IS_DEBUG
    # --USE_CACHED_SCHEMA
end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
echo "Total time taken: $((elapsed_time / 3600)) hours $(( (elapsed_time % 3600) / 60 )) minutes $((elapsed_time % 60)) seconds"