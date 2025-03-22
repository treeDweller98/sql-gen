import argparse
from pathlib import Path
from enum import Enum
import pandas as pd
from core.dbhandler import SQLiteDatabase


class SupoortedModels(Enum):
    qwen25_7b_instruct = "Qwen/Qwen2.5-7B-Instruct"
    qwen25_14b_instruct = "Qwen/Qwen2.5-14B-Instruct"
    qwen25_32b_instruct = "Qwen/Qwen2.5-32B-Instruct"

    qwen25_coder_7b_instruct = "Qwen/Qwen2.5-Coder-7B-Instruct"
    qwen25_coder_14b_instruct = "Qwen/Qwen2.5-Coder-14B-Instruct"
    qwen25_coder_32b_instruct = "Qwen/Qwen2.5-Coder-32B-Instruct"

    granite_32_8b_instruct = "ibm-granite/granite-3.2-8b-instruct"
    granite_8b_code_instruct = "ibm-granite/granite-8b-code-instruct-4k"
    granite_20b_code_instruct = "ibm-granite/granite-20b-code-instruct-8k"
    granite_34b_code_instruct = "ibm-granite/granite-34b-code-instruct-8k"

    deepseek_v2_lite_chat = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    deepseek_coder_v2_lite_instruct = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

    deepseek_r1_qwen_7b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    deepseek_r1_qwen_14b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    deepseek_r1_qwen_32b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    
    qwq_32b = "Qwen/QwQ-32B"


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
    db_names: list[str] = [f.name for f in (input_path / db_foldername).iterdir()]
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
        '--MODEL_MAX_SEQ_LEN', type=int, default=4096 * 2, 
        help="Maximum sequence length for the model (must be integer multiple of 4096)."
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
        help="Dtype for weights and activations. Set to 'half' for cards that don't support BF16"
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

    # Additional parameters like BIRD_QUESTION_FILENAME, etc.
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
        '--USE_CACHED_SCHEMA', type=bool, default=False, 
        help="Flag to use cached schema."
    )

    # Experiment to run
    parser.add_argument(
        '--EXPERIMENT', type=str,
        help="Experiment: zs, rzs, mad, madb."
    )
    parser.add_argument(
        '--IS_DEBUG', type=bool, default=False,
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
    print(f"Model Max Seq Len: {args.MODEL_MAX_SEQ_LEN}")
    print(f"KV Cache Dtype: {args.KV_CACHE_DTYPE}")
    print(f"VLLM_DTYPE: {args.VLLM_DTYPE}")
    print(f"Batch Size: {args.BATCH_SIZE}")
    print(f"Seed: {args.SEED}")
    print(f"Input Path: {args.INPUT_PATH}")
    print(f"Output Path: {args.OUTPUT_PATH}")
    print(f"Bird Question Filename: {args.BIRD_QUESTION_FILENAME}")
    print(f"Databases Folder Name: {args.DB_FOLDERNAME}")
    print(f"DB Exec Timeout: {args.DB_EXEC_TIMEOUT}")
    print(f"Use Cached Schema: {args.USE_CACHED_SCHEMA}")
    print("\n\n")
    return args


if __name__ == '__main__':
    args = parse_args()
