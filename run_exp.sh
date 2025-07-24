source ".venv/bin/activate"
source "secrets.env"
huggingface-cli login --token $HF_TOKEN
wandb login --relogin $WANDB_API_KEY

experiment="zs"     # zs, rzs, mad, pick, plan, plan-exec
dataset="spider"    # spider, bird
seed=42             # 42, 118, 98
tensor_parallel=1

zs_models=(
    "qwen25_7b_instruct"
    "qwen25_14b_instruct"
    "qwen25_32b_instruct"
    "qwen25_coder_7b_instruct"
    "qwen25_coder_14b_instruct"
    "qwen25_coder_32b_instruct"
    "gemma3_4b_it"
    "gemma3_12b_it"
    "gemma3_27b_it"
    "codellama_7b_instruct"
    "codellama_13b_instruct"
    "codellama_34b_instruct"
    "granite_32_8b_instruct"
    "granite_33_8b_instruct"
    "deepseek_v2_lite_chat"
    "deepseek_coder_v2_lite_instruct"
    "ministral_8b_instruct"
    "mistral_small_24b_instruct"
    "llama3_8b_instruct"
    "llama31_8b_instruct"
    "starcoder_15b_instruct"
)
rzs_models=(
    "deepseek_r1_qwen_7b"
    "deepseek_r1_qwen_14b"
    "deepseek_r1_qwen_32b"
    "qwq_32b"
)


if [[ "$experiment" == "zs" || "$experiment" == "mad" || "$experiment" == "plan-exec" ]]; then
    models=("${zs_models[@]}")
elif [[ "$experiment" == "rzs" || "$experiment" == "pick" || "$experiment" == "plan" ]]; then
    models=("${rzs_models[@]}")
else
    echo "Unknown experiment type: $experiment"
    exit 1
fi

if [ "$dataset" == "spider" ]; then
    input_dir="data/spider_data"
    db_foldername="database"
    question_filename="dev.json"
else
    input_dir="data/bird_minidev"
    db_foldername="dev_databases"
    question_filename="dev.json"
fi


for model in "${models[@]}"; do
    output_dir="results_${dataset}/${experiment}/${model}_${experiment}/"
    mkdir -p "$output_dir"
    python main.py \
        --EXPERIMENT "$experiment" \
        --MODEL "$model" \
        --GPU_MEMORY_UTILIZATION "0.97" \
        --TENSOR_PARALLEL_SIZE "$tensor_parallel" \
        --MAX_MODEL_LEN "8192" \
        --DISTRIBUTED_EXECUTOR_BACKEND 'mp' \
        --KV_CACHE_DTYPE "auto" \
        --VLLM_DTYPE "auto" \
        --ENFORCE_EAGER \
        --ENABLE_PREFIX_CACHING \
        --SEED "$seed" \
        --BATCH_SIZE "256" \
        --DATASET "$dataset" \
        --INPUT_DIR "$input_dir" \
        --OUTPUT_DIR "$output_dir" \
        --QUESTION_FILENAME "$question_filename" \
        --DB_FOLDERNAME "$db_foldername" \
        --DB_EXEC_TIMEOUT "30" \
        > "${output_dir}/log.out" 2> "${output_dir}/log.err"
done