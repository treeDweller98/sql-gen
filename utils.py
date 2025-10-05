import argparse
import contextlib
import gc
from enum import Enum
from pathlib import Path

import pandas as pd
import ray
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

from core.dbhandler import SQLiteDatabase


class SupportedModels(Enum):
    codellama_7b_instruct = "codellama/CodeLlama-7b-Instruct-hf"
    codellama_13b_instruct = "codellama/CodeLlama-13b-Instruct-hf"
    codellama_34b_instruct = "codellama/CodeLlama-34b-Instruct-hf"

    llama3_8b_instruct = "meta-llama/Meta-Llama-3-8B-Instruct"
    llama31_8b_instruct = "meta-llama/Llama-3.1-8B-Instruct"

    ministral_8b_instruct = "mistralai/Ministral-8B-Instruct-2410"
    mistral_small_24b_instruct = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    codegemma_7b_it = "google/codegemma-7b-it"

    gemma3_4b_it = "google/gemma-3-4b-it"
    gemma3_12b_it = "google/gemma-3-12b-it"
    gemma3_27b_it = "google/gemma-3-27b-it"

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

    starcoder_15b_instruct = "bigcode/starcoder2-15b-instruct-v0.1"

    deepseek_r1_qwen_7b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    deepseek_r1_qwen_14b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    deepseek_r1_qwen_32b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    qwq_32b = "Qwen/QwQ-32B"


def load_llm(args) -> LLM:
    return LLM(
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


def del_llm(llm: LLM) -> None:
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully delete the llm pipeline and free the GPU memory.")


### Dataset Reader Functions ###
def read_bird_dataset(
    input_dir: Path,
    question_filename: str,
    db_foldername: str,
    use_cached_schema: Path | None,
    db_exec_timeout: float,
    is_debug: bool = False,
) -> tuple[pd.DataFrame, dict[str, SQLiteDatabase]]:
    """BIRD dataset reader function.
    1. Reads dataset into DataFrame from "input_dir/question_filename".
    2. Lists database names from folders in "input_dir/db_foldername/".
    3. Creates dict of SQLiteDatabases, indexed by db_name.
    Returns DataFrame of BIRD questions and dict of databases.
    """
    df = pd.read_json(input_dir / question_filename)
    df.rename(columns={"SQL": "gold_sql"}, inplace=True)
    df["question"] = df.apply(
        lambda row: f"{row['question']}  Note: {row['evidence']}",
        axis=1,
    )
    db_names: list[str] = [f.name for f in (input_dir / db_foldername).iterdir() if f.is_dir()]
    databases: dict[str, SQLiteDatabase] = {
        db_id: SQLiteDatabase(db_id, (input_dir / db_foldername), db_exec_timeout, use_cached_schema)
        for db_id in db_names
    }
    if is_debug:
        df = df.head().reset_index()
    print(f"\n\n{db_names=}\n{len(df)=}\n\n")
    return df, databases


def read_spider_dataset(
    input_dir: Path,
    question_filename: str,
    db_foldername: str,
    use_cached_schema: Path | None,
    db_exec_timeout: float,
    is_debug: bool = False,
):
    df = pd.read_json(input_dir / question_filename)
    df.rename(columns={"query": "gold_sql"}, inplace=True)
    df["question_id"] = range(len(df))
    df["difficulty"] = "spider"
    db_names: list[str] = [f.name for f in (input_dir / db_foldername).iterdir() if f.is_dir()]
    databases: dict[str, SQLiteDatabase] = {
        db_id: SQLiteDatabase(db_id, (input_dir / db_foldername), db_exec_timeout, use_cached_schema)
        for db_id in db_names
    }
    if is_debug:
        df = df.head().reset_index()
    print(f"\n\n{db_names=}\n{len(df)=}\n\n")
    return df, databases


### ARG PARSER
def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for Experiment.")

    # Model argument
    model_choices = [model.name for model in SupportedModels]
    parser.add_argument(
        "--MODEL",
        type=str,
        choices=model_choices,
        help=f"Model to be used in the experiment: {model_choices}",
    )
    parser.add_argument(
        "--TENSOR_PARALLEL_SIZE",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--MAX_MODEL_LEN",
        type=int,
        default=4096 * 2,
        help="Maximum length of the model (must be integer multiple of 4096).",
    )
    parser.add_argument(
        "--KV_CACHE_DTYPE",
        type=str,
        default="auto",
        choices=["auto", "fp8"],
        help="KV cache data type (auto/fp8).",
    )
    parser.add_argument(
        "--GPU_MEMORY_UTILIZATION",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0 to 1.0).",
    )
    parser.add_argument(
        "--VLLM_DTYPE",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="Dtype for weights and activations. Set to 'half' for cards that don't support BF16",
    )
    parser.add_argument(
        "--DISTRIBUTED_EXECUTOR_BACKEND",
        type=str,
        default="ray",
        choices=["ray", "mp", "uni", "external_launcher"],
        help="ParallelConfig option. Setting to 'mp' allows unloading.",
    )
    parser.add_argument(
        "--ENABLE_PREFIX_CACHING",
        action="store_true",
        help="VLLM enables prompt prefix caching",
    )
    parser.add_argument("--ENFORCE_EAGER", action="store_true", help="VLLM uses PyTorch eager mode")
    parser.add_argument("--SEED", type=int, default=42, help="Seed for random number generation.")

    # Save once a batch is completed
    parser.add_argument(
        "--BATCH_SIZE",
        type=int,
        default=64,
        help="Number of examples sent for generation at once. Saves after each batch is finished.",
    )

    # Input and Output Path
    parser.add_argument("--DATASET", type=str, default="bird", help="Dataset: 'spider' or 'bird'.")
    parser.add_argument(
        "--INPUT_DIR",
        type=str,
        default="data/bird-minidev",
        help="Directory for the experiment data.",
    )
    parser.add_argument(
        "--OUTPUT_DIR",
        type=str,
        default="results/{model}_{experiment}/",
        help="Output path for the experiment results.",
    )
    parser.add_argument(
        "--QUESTION_FILENAME",
        type=str,
        default="dev.json",
        help="Question file in input_dir/",
    )
    parser.add_argument(
        "--DB_FOLDERNAME",
        type=str,
        default="dev_databases",
        help="Folder name for databases in in input_dir/",
    )
    parser.add_argument(
        "--DB_EXEC_TIMEOUT",
        type=float,
        default=30.0,
        help="Timeout for database queries in seconds.",
    )
    parser.add_argument(
        "--USE_CACHED_SCHEMA",
        type=str,
        default="",
        help="Optional cached schema JSON filepath.",
    )

    # Experiment to run
    parser.add_argument(
        "--EXPERIMENT",
        type=str,
        help="Experiment: zs, rzs, mad, madb, pick, plan, plan-exec",
    )
    parser.add_argument("--IS_DEBUG", action="store_true", help="If debug, runs with first 5 rows of df")

    args = parser.parse_args()
    args.MODEL = SupportedModels[args.MODEL]
    args.INPUT_DIR = Path(args.INPUT_DIR)
    args.OUTPUT_DIR = Path(args.OUTPUT_DIR)
    args.USE_CACHED_SCHEMA = Path(args.USE_CACHED_SCHEMA) if args.USE_CACHED_SCHEMA else None

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

    print(f"Dataset: {args.DATASET}")
    print(f"Input Path: {args.INPUT_DIR}")
    print(f"Output Path: {args.OUTPUT_DIR}")
    print(f"Databases Folder Name: {args.DB_FOLDERNAME}")
    print(f"Question Filename: {args.QUESTION_FILENAME}")
    print(f"DB Exec Timeout: {args.DB_EXEC_TIMEOUT}")
    print(f"Use Cached Schema: {args.USE_CACHED_SCHEMA}")
    print(f"Batch Size: {args.BATCH_SIZE}")
    print("\n\n")
    return args


if __name__ == "__main__":
    args = parse_args()
    df, databases = read_bird_dataset(
        input_dir=args.INPUT_DIR,
        question_filename=args.QUESTION_FILENAME,
        db_foldername=args.DB_FOLDERNAME,
        use_cached_schema=args.USE_CACHED_SCHEMA,
        db_exec_timeout=args.DB_EXEC_TIMEOUT,
        is_debug=args.IS_DEBUG,
    )
