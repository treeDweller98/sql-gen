import gc
import contextlib
import argparse
from pathlib import Path
from enum import Enum
import pandas as pd
import ray
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from core.dbhandler import SQLiteDatabase


class SupoortedModels(Enum):
    codellama_7b_instruct  = "codellama/CodeLlama-7b-Instruct-hf"
    codellama_13b_instruct = "codellama/CodeLlama-13b-Instruct-hf"
    codellama_34b_instruct = "codellama/CodeLlama-34b-Instruct-hf"

    codegemma_7b_it = "google/codegemma-7b-it"
    
    gemma3_4b_it  = "google/gemma-3-4b-it"
    gemma3_12b_it = "google/gemma-3-12b-it"
    gemma3_27b_it = "google/gemma-3-27b-it"

    qwen25_1_5b = "Qwen/Qwen2.5-1.5B"
    qwen25_3b = "Qwen/Qwen2.5-3B"
    qwen25_7b = "Qwen/Qwen2.5-7B"

    qwen25_1_5b_instruct = "Qwen/Qwen2.5-1.5B-Instruct"
    qwen25_3b_instruct = "Qwen/Qwen2.5-3B-Instruct"
    qwen25_7b_instruct = "Qwen/Qwen2.5-7B-Instruct"
    qwen25_14b_instruct = "Qwen/Qwen2.5-14B-Instruct"
    qwen25_32b_instruct = "Qwen/Qwen2.5-32B-Instruct"

    qwen25_coder_1_5b_instruct = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    qwen25_coder_3b_instruct = "Qwen/Qwen2.5-Coder-3B-Instruct"
    qwen25_coder_7b_instruct = "Qwen/Qwen2.5-Coder-7B-Instruct"
    qwen25_coder_14b_instruct = "Qwen/Qwen2.5-Coder-14B-Instruct"
    qwen25_coder_32b_instruct = "Qwen/Qwen2.5-Coder-32B-Instruct"

    granite_3b_code_instruct_128k = "ibm-granite/granite-3b-code-instruct-128k"
    granite_8b_code_instruct_128k = "ibm-granite/granite-8b-code-instruct-128k"
    granite_20b_code_instruct_8k = "ibm-granite/granite-20b-code-instruct-8k"
    granite_34b_code_instruct_8k = "ibm-granite/granite-34b-code-instruct-8k"

    granite_32_2b_instruct = "ibm-granite/granite-3.2-2b-instruct"
    granite_32_8b_instruct = "ibm-granite/granite-3.2-8b-instruct"
    granite_33_2b_instruct = "ibm-granite/granite-3.3-2b-instruct"
    granite_33_8b_instruct = "ibm-granite/granite-3.3-8b-instruct"

    deepseek_v2_lite_chat = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    deepseek_coder_v2_lite_instruct = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

    starcoder_15b_instruct = "bigcode/starcoder2-15b-instruct-v0.1"     # turn off prefix-caching

    deepseek_r1_qwen_7b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    deepseek_r1_qwen_14b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    deepseek_r1_qwen_32b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    
    qwq_32b = "Qwen/QwQ-32B"


def load_llm(args) -> LLM:
    llm = LLM(
        args.MODEL.value,
        max_model_len=args.MAX_MODEL_LEN,
        gpu_memory_utilization=args.GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=args.TENSOR_PARALLEL_SIZE,
        kv_cache_dtype=args.KV_CACHE_DTYPE,
        enable_prefix_caching=args.ENABLE_PREFIX_CACHING,
        enforce_eager=args.ENFORCE_EAGER,
        dtype=args.VLLM_DTYPE,
        distributed_executor_backend=args.DISTRIBUTED_EXECUTOR_BACKEND,
        seed=args.SEED,
        trust_remote_code=True,
    )
    return llm

def del_llm(llm: LLM) -> None:
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully delete the llm pipeline and free the GPU memory.")


### BIRD Dataset Reader Function ###
def read_dataset(
    input_path: Path, bird_question_filename: str, db_foldername: str, 
    use_cached_schema: bool, db_exec_timeout: float, is_debug: bool = False,
) -> tuple[pd.DataFrame, dict[str, SQLiteDatabase]]:
    """ BIRD dataset reader function.
        1. Reads dataset into DataFrame from "input_path/bird_question_filename".
        2. Lists database names from folders in "input_path/db_foldername/".
        3. Creates dict of SQLiteDatabases, indexed by db_name.
        Returns DataFrame of BIRD questions and dict of databases.
    """
    df = pd.read_json(input_path / bird_question_filename)
    df.rename(columns={'SQL': 'gold_sql'}, inplace=True)
    db_names: list[str] = [f.name for f in (input_path / db_foldername).iterdir() if f.is_dir()]
    databases: dict[str, SQLiteDatabase] = {
        db_id: SQLiteDatabase(db_id, (input_path / db_foldername), db_exec_timeout, use_cached_schema) 
        for db_id in db_names
    }
    if is_debug:
        df = df.head().reset_index()
    print(f'\n\n{db_names=}\n{len(df)=}\n\n')
    return df, databases


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for Experiment.")
    
    # Model argument
    model_choices = [model.name for model in SupoortedModels]
    parser.add_argument(
        '--MODEL', type=str, choices=model_choices, 
        help=f"Model to be used in the experiment: {model_choices}"
    )
    parser.add_argument(
        '--TENSOR_PARALLEL_SIZE', type=int, default=1,
        help="Number of GPUs for tensor parallelism."
    )
    parser.add_argument(
        '--MAX_MODEL_LEN', type=int, default=4096 * 2, 
        help="Maximum length of the model (must be integer multiple of 4096)."
    )
    parser.add_argument(
        '--KV_CACHE_DTYPE', type=str, default='auto', choices=['auto', 'fp8'],
        help="KV cache data type (auto/fp8)."
    )
    parser.add_argument(
        '--GPU_MEMORY_UTILIZATION', type=float, default=0.9,
        help="GPU memory utilization (0.0 to 1.0)."
    )
    parser.add_argument(
        '--VLLM_DTYPE', type=str, default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help="Dtype for weights and activations. Set to 'half' for cards that don't support BF16"
    )
    parser.add_argument(
        '--DISTRIBUTED_EXECUTOR_BACKEND', type=str, default='ray',
        choices= ['ray', 'mp', 'uni', 'external_launcher'],
        help="ParallelConfig option. Setting to 'mp' allows unloading."
    )
    parser.add_argument(
        '--ENABLE_PREFIX_CACHING', action='store_true',
        help="VLLM enables prompt prefix caching"
    )
    parser.add_argument(
        '--ENFORCE_EAGER', action='store_true',
        help="VLLM uses PyTorch eager mode"
    )
    parser.add_argument(
        '--SEED', type=int, default=42, 
        help="Seed for random number generation."
    )
    
    # Save once a batch is completed
    parser.add_argument(
        '--BATCH_SIZE', type=int, default=64, 
        help="Number of examples sent for generation at once. Saves after each batch is finished."
    )

    # Input and Output Path
    parser.add_argument(
        '--INPUT_PATH', type=str, default='data/bird-minidev',
        help="Input path for the experiment data."
    )
    parser.add_argument(
        '--OUTPUT_PATH', type=str, default='results/{model}_{experiment}/', 
        help="Output path for the experiment results."
    )
    parser.add_argument(
        '--BIRD_QUESTION_FILENAME', type=str, default='dev.json', 
        help="Filename for bird question data in input_path/"
    )
    parser.add_argument(
        '--DB_FOLDERNAME', type=str, default='dev_databases', 
        help="Folder name for databases in in input_path/"
    )
    parser.add_argument(
        '--DB_EXEC_TIMEOUT', type=float, default=30.0, 
        help="Timeout for database queries in seconds."
    )
    parser.add_argument(
        '--USE_CACHED_SCHEMA', type=str, default='', 
        help="Optional cached schema JSON filepath."
    )

    # Experiment to run
    parser.add_argument(
        '--EXPERIMENT', type=str,
        help="Experiment: zs, rzs, mad, madb ,grpo"
    )
    parser.add_argument(
        '--IS_DEBUG', action='store_true',
        help="If debug, runs with first 5 rows of df"
    )

    args = parser.parse_args()
    args.MODEL = SupoortedModels[args.MODEL]
    args.INPUT_PATH = Path(args.INPUT_PATH)
    args.OUTPUT_PATH = Path(args.OUTPUT_PATH.format(model=args.MODEL.name, experiment=args.EXPERIMENT))

    # Print the configurations to confirm
    print(f"Experiment: {args.EXPERIMENT}")
    print(f"Debug mode: {args.IS_DEBUG}")
    print(f"Model: {args.MODEL.value}")

    print(f"GPU Memory Utilization: {args.GPU_MEMORY_UTILIZATION}")
    print(f"Tensor Parallel Size: {args.TENSOR_PARALLEL_SIZE}")
    print(f"Max Model Len: {args.MAX_MODEL_LEN}")
    print(f"KV Cache Dtype: {args.KV_CACHE_DTYPE}")
    print(f"VLLM dtype: {args.VLLM_DTYPE}")
    print(f"Enable Prefix Caching: {args.ENABLE_PREFIX_CACHING}")
    print(f"Enforce Eager: {args.ENFORCE_EAGER}")
    print(f"Distributed Executor Backend: {args.DISTRIBUTED_EXECUTOR_BACKEND}")
    print(f"Seed: {args.SEED}")

    print(f"Input Path: {args.INPUT_PATH}")
    print(f"Output Path: {args.OUTPUT_PATH}")
    print(f"Databases Folder Name: {args.DB_FOLDERNAME}")
    print(f"Bird Question Filename: {args.BIRD_QUESTION_FILENAME}")
    print(f"DB Exec Timeout: {args.DB_EXEC_TIMEOUT}")
    print(f"Use Cached Schema: {args.USE_CACHED_SCHEMA}")
    print(f"Batch Size: {args.BATCH_SIZE}")
    print("\n\n")
    return args


if __name__ == '__main__':
    args = parse_args()
    df, databases = read_dataset(
        input_path=args.INPUT_PATH,
        bird_question_filename=args.BIRD_QUESTION_FILENAME,
        db_foldername=args.DB_FOLDERNAME,
        use_cached_schema=args.USE_CACHED_SCHEMA,
        db_exec_timeout=args.DB_EXEC_TIMEOUT,
        is_debug=args.IS_DEBUG
    )

# python utils.py \
#     --EXPERIMENT "rzs" \
#     --MODEL qwen25_coder_14b_instruct \
#     --GPU_MEMORY_UTILIZATION 0.97 \
#     --TENSOR_PARALLEL_SIZE 4 \
#     --KV_CACHE_DTYPE "auto" \
#     --VLLM_DTYPE "auto" \
#     --MAX_MODEL_LEN 8192 \
#     --ENFORCE_EAGER \
#     --ENABLE_PREFIX_CACHING \
#     --SEED 42 \
#     --BATCH_SIZE 128 \
#     --INPUT_PATH "data/bird-minidev" \
#     --OUTPUT_PATH "results/rzs/{model}_{experiment}/" \
#     --BIRD_QUESTION_FILENAME "dev.json" \
#     --DB_FOLDERNAME "dev_databases" \
#     --DB_EXEC_TIMEOUT 30
#     --IS_DEBUG
#     --DISTRIBUTED_EXECUTOR_BACKEND 'ray'
#     --USE_CACHED_SCHEMA