#!/bin/bash

#SBATCH --array=0-3
#SBATCH --partition=long
#SBATCH --time=3:00:00
#SBATCH -o log/rplan_minidev/log-%A-%a.out
#SBATCH -e log/rplan_minidev/log-%A-%a.err
#SBATCH -J rl-llm-rplan
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=48G

module load cuda/12.6.0
source ~/scratch/ria-vlm-abs-rl/rl-llm/sql-gen/.venv-sql/bin/activate

models_list=(
    "deepseek_r1_qwen_7b"
    "deepseek_r1_qwen_14b"
    "deepseek_r1_qwen_32b"
    "qwq_32b"
)
model=${models_list[$SLURM_ARRAY_TASK_ID]}

start_time=$(date +%s)
python main.py \
    --EXPERIMENT "plan" \
    --MODEL $model \
    --GPU_MEMORY_UTILIZATION 0.97 \
    --TENSOR_PARALLEL_SIZE 2 \
    --MAX_MODEL_LEN 16384 \
    --DISTRIBUTED_EXECUTOR_BACKEND 'mp' \
    --KV_CACHE_DTYPE "auto" \
    --VLLM_DTYPE "auto" \
    --ENFORCE_EAGER \
    --ENABLE_PREFIX_CACHING \
    --SEED 42 \
    --BATCH_SIZE 256 \
    --INPUT_PATH "data/bird-minidev" \
    --OUTPUT_PATH "results/rplan/{model}_{experiment}/" \
    --BIRD_QUESTION_FILENAME "dev.json" \
    --DB_FOLDERNAME "dev_databases" \
    --DB_EXEC_TIMEOUT 30
    # --IS_DEBUG
    # --USE_CACHED_SCHEMA
end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
echo "Total time taken: $((elapsed_time / 3600)) hours $(( (elapsed_time % 3600) / 60 )) minutes $((elapsed_time % 60)) seconds"