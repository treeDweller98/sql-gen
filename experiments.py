from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams
from core.birdeval import evaluate
from core.dbhandler import SQLiteDatabase
from sqlgen.base_agent import TextToSQL
from sqlgen.discussion import MultiAgentDiscussion
from sqlgen.zeroshot import ZeroShotAgent
from typing import Callable


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
    outputs, labels = agent_zs.batched_generate(
        df, cfg, batch_size, 
        savename, evaluate
    )