from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams
from core.birdeval import evaluate
from core.dbhandler import SQLiteDatabase
from sqlgen.base_agent import TextToSQL
from sqlgen.discussion import MultiAgentDiscussion
from sqlgen.zeroshot import ZeroShotAgent


def run_baseline(
    agent: TextToSQL, cfg: SamplingParams, df: pd.DataFrame, 
    batch_size: int, output_path: Path, savename: str, evaluator_fn: Callable, **kwargs
) -> tuple[pd.DataFrame, str]:
    outputs, labels = agent.batched_generate(df, cfg, batch_size, savename, evaluator_fn, **kwargs)

    df[f'input_prompts_{savename}'] = outputs.input_prompts
    df[f'n_in_tokens_{savename}']   = outputs.n_in_tokens
    df[f'raw_responses_{savename}'] = outputs.raw_responses
    df[f'n_out_tokens_{savename}']  = outputs.n_out_tokens
    df[f'parsed_sql_{savename}']    = outputs.parsed_sql    
    df[f'label_{savename}']         = labels

    df.to_json(output_path/f'df_{savename}.json', orient='records')
    return df


def discuss_experiment(
    df: pd.DataFrame, databases: dict[str, SQLiteDatabase], llm: LLM, 
    output_path: Path, batch_size: int, savename: str = f'multiag'
):
    MultiAgentDiscussion.discuss(
        df=df, 
        databases=databases, 
        llm=llm, 
        output_path=output_path, 
        savename=savename, 
        batch_size=batch_size, 
        evaluator_fn=evaluate, 
    )


def zeroshot_experiment(
    df: pd.DataFrame, databases: dict[str, SQLiteDatabase], 
    llm: LLM, cfg: SamplingParams, 
    output_path: Path, batch_size: int, savename: str = f'zs'
):  
    agent_zs = ZeroShotAgent(llm, databases, output_path)
    df = run_baseline(agent_zs, cfg, df, batch_size, output_path, savename, evaluate)