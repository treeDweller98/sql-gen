from typing import Callable
import pandas as pd
from vllm import LLM, SamplingParams
from config import *
from core.dbhandler import SQLiteDatabase
from core.birdeval import evaluate
from sqlgen.base_agent import TextToSQL
from sqlgen.discussion import MultiAgentDiscussion
from sqlgen.zeroshot import ZeroShotAgent


### BIRD Dataset Reader Function ###
def read_dataset() -> tuple[pd.DataFrame, dict[str, SQLiteDatabase]]:
    """ BIRD dataset reader function.
        1. Reads dataset into DataFrame from "INPUT_PATH/BIRD_QUESTION_FILENAME".
        2. Lists database names from folders in "INPUT_PATH/DB_FOLDERNAME/".
        3. Creates dict of SQLiteDatabases, indexed by db_name.
        Returns DataFrame of BIRD questions and dict of databases.
    """
    df = pd.read_json(INPUT_PATH / BIRD_QUESTION_FILENAME)
    db_names: list[str] = [f.name for f in (INPUT_PATH / DATABASES_FOLDERNAME).iterdir()]
    databases: dict[str, SQLiteDatabase] = {
        db_id: SQLiteDatabase(db_id, (INPUT_PATH / DATABASES_FOLDERNAME), DB_EXEC_TIMEOUT, USE_CACHED_SCHEMA) 
        for db_id in db_names
    }
    print(f'{db_names=}, {len(df)=}')
    return df, databases


def setup_experiment():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    df, databases = read_dataset()
    cfg = SamplingParams(
        temperature=0,
        top_p=1,
        repetition_penalty=1.1,
        max_tokens=4096,
    )
    llm = LLM(
        MODEL.value,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MODEL_MAX_SEQ_LEN,
        max_seq_len_to_capture=MODEL_MAX_SEQ_LEN,
        kv_cache_dtype=KV_CACHE_DTYPE,
        seed=SEED,
    )
    return df, databases, cfg, llm


def agent_baseline(
    agent: TextToSQL, cfg: SamplingParams, df: pd.DataFrame, 
    batch_size: int, savename: str, evaluator_fn: Callable, **kwargs
) -> tuple[pd.DataFrame, str]:
    print(f"Experiment: {savename}_{'' if USE_CACHED_SCHEMA else 'un'}aug_{MODEL.name}")
    outputs, labels = agent.batched_generate(df, cfg, batch_size, savename, evaluator_fn, **kwargs)

    df[f'input_prompts_{savename}'] = outputs.input_prompts
    df[f'n_in_tokens_{savename}']   = outputs.n_in_tokens
    df[f'raw_responses_{savename}'] = outputs.raw_responses
    df[f'n_out_tokens_{savename}']  = outputs.n_out_tokens
    df[f'parsed_sql_{savename}']    = outputs.parsed_sql    
    df[f'label_{savename}']         = labels
    df.to_json(OUTPUT_PATH/f'df_{savename}.json', orient='records')
        
    print(f"Experiment: {savename}_{'' if USE_CACHED_SCHEMA else 'un'}aug_{MODEL.name}_{EXPERIMENT} Successfully Completed.\n\n\n")
    return df


def mad_experiment(
    df: pd.DataFrame, databases: dict[str, SQLiteDatabase], llm: LLM, savename: str = f'multiag'
):
    MultiAgentDiscussion.discuss(
        df=df, 
        databases=databases, 
        llm=llm, 
        output_path=OUTPUT_PATH, 
        savename=savename, 
        batch_size=BATCH_SIZE, 
        evaluator_fn=evaluate, 
    )


def zeroshot_experiment(
    df: pd.DataFrame, databases: dict[str, SQLiteDatabase], 
    llm: LLM, cfg: SamplingParams, savename: str = f'zs'
):  
    agent_zs = ZeroShotAgent(llm, databases, OUTPUT_PATH)
    df = agent_baseline(agent_zs, cfg, df, BATCH_SIZE, savename, evaluate)