import pandas as pd
from vllm import LLM, SamplingParams
from config import *
from core.dbhandler import SQLiteDatabase
from core.birdeval import evaluate
from sqlgen.base_agent import TextToSQL


### BIRD Dataset Reader Function ###
def read_dataset() -> tuple[pd.DataFrame, list[str], dict[str, SQLiteDatabase]]:
    """ BIRD dataset reader function.
        1. Reads dataset into DataFrame from "INPUT_PATH/BIRD_QUESTION_FILENAME".
        2. Lists database names from folders in "INPUT_PATH/DB_FOLDERNAME/".
        3. If USE_DEBUG_DB, returns debug subset of databases
           ['formula_1', 'debit_card_specializing', 'thrombosis_prediction'].
        4. If USE_DEBUG_DATASET, returns first 5 DataFrame rows only.
        5. Creates dict of SQLiteDatabases, indexed by db_name.
        Returns df of BIRD questions, db_names, and dict of databases.
    """
    df = pd.read_json(INPUT_PATH / BIRD_QUESTION_FILENAME)
    if USE_DEBUG_DB:
        db_names: list[str] = ['formula_1', 'debit_card_specializing', 'thrombosis_prediction']
        df = df[df['db_id'].isin(db_names)]
    else:
        db_names: list[str] = [f.name for f in (INPUT_PATH / DATABASES_FOLDERNAME).iterdir()]
    if USE_DEBUG_DATASET:
        df = df.head()
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
    batch_size: int, name: str, **kwargs
) -> tuple[pd.DataFrame, str]:
    
    print(f"Experiment: {name}_{'' if USE_CACHED_SCHEMA else 'un'}aug_{MODEL.name}_{EXPERIMENT}")
    
    outputs = agent.batched_generate(df, cfg, batch_size, name, **kwargs)

    df[f'input_prompts_{name}'] = outputs.input_prompts
    df[f'n_in_tokens_{name}']   = outputs.n_in_tokens
    df[f'raw_responses_{name}'] = outputs.raw_responses
    df[f'n_out_tokens_{name}']  = outputs.n_out_tokens
    df[f'parsed_sql_{name}']    = outputs.parsed_sql
    
    labels, report = evaluate(df, agent.databases, DB_EXEC_TIMEOUT, f'parsed_sql_{name}')
    df[f'label_{name}'] = labels
    
    with open(OUTPUT_PATH/f'results_{name}.txt', 'w') as f:
        f.write(report)
    df.to_json(OUTPUT_PATH/f'df_{name}.json', orient='records')
    
    print(f"Experiment: {name}_{'' if USE_CACHED_SCHEMA else 'un'}aug_{MODEL.name}_{EXPERIMENT} Successfully Completed.\n\n\n")
    return df


def mad_experiment(
    df: pd.DataFrame, databases: dict[str, SQLiteDatabase], llm: LLM, savename: str = f'multiag'
):
    from sqlgen.discussion import MultiAgentDiscussion
    MultiAgentDiscussion.discuss(df, databases, llm, OUTPUT_PATH, savename, BATCH_SIZE, evaluate, DB_EXEC_TIMEOUT)



# from sqlgen.zeroshot import ZeroShotAgent, MetaPromptZeroShotAgent, OptimizerAgent

# def run_baseline(df, databases, cfg, llm, output_path, batch_size):
    
#     agent_zs = ZeroShotAgent(llm, databases, output_path)
#     agent_mp = MetaPromptZeroShotAgent(llm, databases, output_path)
#     agent_op = OptimizerAgent(llm, databases, output_path)

#     df = agent_baseline(agent_zs, cfg, df, batch_size, 'zs')
#     df = agent_baseline(agent_op, cfg, df, batch_size, 'opzs', pred_col='pred_zs')
#     df = agent_baseline(agent_mp, cfg, df, batch_size, 'mp')
#     df = agent_baseline(agent_op, cfg, df, batch_size, 'opmp', pred_col='pred_mp')

#     df.to_json(output_path / 'final_df.json', orient='records')