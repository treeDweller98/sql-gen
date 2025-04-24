#!/bin/bash

#SBATCH --array=0-7
#SBATCH --partition=long
#SBATCH --time=36:00:00
#SBATCH -o log/mad_minidev/log-%A-%a.out
#SBATCH -e log/mad_minidev/log-%A-%a.err 
#SBATCH -J rl-llm
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=48G

source ~/scratch/ria-vlm-abs-rl/rl-llm/sql-gen/.venv/bin/activate

models_list=(
    "qwen25_3b_instruct"
    "qwen25_coder_3b_instruct"
    "deepseek_coder_v2_lite_instruct"
    "granite_32_2b_instruct"
    "granite_3b_code_instruct_2k"
    "granite_8b_code_instruct_4k"
    "granite_34b_code_instruct_8k"
    "granite_20b_code_instruct_8k"
)
model_lens=(
    8192
    8192
    8192
    8192
    2048
    4096
    8192
    8192
)
model=${models_list[$SLURM_ARRAY_TASK_ID]}
model_len=${model_lens[$SLURM_ARRAY_TASK_ID]}

start_time=$(date +%s)
python main.py \
    --EXPERIMENT "mad" \
    --MODEL $model \
    --GPU_MEMORY_UTILIZATION 0.97 \
    --TENSOR_PARALLEL_SIZE 2 \
    --MODEL_MAX_SEQ_LEN $model_len \
    --DISTRIBUTED_EXECUTOR_BACKEND 'mp' \
    --KV_CACHE_DTYPE "auto" \
    --VLLM_DTYPE "auto" \
    --ENFORCE_EAGER \
    --ENABLE_PREFIX_CACHING \
    --SEED 42 \
    --BATCH_SIZE 128 \
    --INPUT_PATH "data/bird-minidev" \
    --OUTPUT_PATH "results/mad/{model}_{experiment}/" \
    --BIRD_QUESTION_FILENAME "dev.json" \
    --DB_FOLDERNAME "dev_databases" \
    --DB_EXEC_TIMEOUT 30
    # --IS_DEBUG
    # --USE_CACHED_SCHEMA
end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
echo "Total time taken: $((elapsed_time / 3600)) hours $(( (elapsed_time % 3600) / 60 )) minutes $((elapsed_time % 60)) seconds"