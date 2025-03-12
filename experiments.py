from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams
from core.birdeval import evaluate
from core.dbhandler import SQLiteDatabase
from sqlgen.zeroshot import ZeroShotAgent
from sqlgen.reason import ReasonerZeroShot
from sqlgen.discussion import MultiAgentDiscussion
from sqlgen.debate import MultiAgentDebate
from sqlgen.picker import ReasonerPicker

def zeroshot_experiment(
    df: pd.DataFrame, databases: dict[str, SQLiteDatabase], 
    llm: LLM, cfg: SamplingParams, 
    output_path: Path, batch_size: int, savename: str = 'zs'
):  
    agent_zs = ZeroShotAgent(llm, databases, output_path)
    outputs, labels = agent_zs.batched_generate(
        df, cfg, batch_size, 
        savename, evaluate
    )


def reasoner_zeroshot_experiment(
    df: pd.DataFrame, databases: dict[str, SQLiteDatabase], 
    llm: LLM, cfg: SamplingParams, 
    output_path: Path, batch_size: int, savename: str = 'rzs'
):  
    agent_rzs = ReasonerZeroShot(llm, databases, output_path)
    outputs, labels = agent_rzs.batched_generate(
        df, cfg, batch_size, 
        savename, evaluate
    )


def discuss_experiment(
    df: pd.DataFrame, databases: dict[str, SQLiteDatabase], 
    llm: LLM, cfg: SamplingParams,
    output_path: Path, batch_size: int, savename: str = 'mad'
):
    MultiAgentDiscussion.discuss(
        df=df, 
        databases=databases, 
        llm=llm,
        cfg=cfg,
        output_path=output_path, 
        batch_size=batch_size,
        savename=savename, 
        evaluator_fn=evaluate, 
    )


def debate_experiment(
    df: pd.DataFrame, databases: dict[str, SQLiteDatabase], 
    llm: LLM, cfg: SamplingParams,
    output_path: Path, batch_size: int, savename: str = 'madb'
):
    MultiAgentDebate.debate(
        df=df, 
        databases=databases, 
        llm=llm,
        cfg=cfg,
        output_path=output_path, 
        batch_size=batch_size,
        savename=savename, 
        evaluator_fn=evaluate, 
    )

def reasoner_picker_experiment(
    df: pd.DataFrame, databases: dict[str, SQLiteDatabase], 
    llm: LLM, cfg: SamplingParams,
    output_path: Path, batch_size: int, savename: str = 'pick'
):
    ReasonerPicker.run(
        df=df, 
        databases=databases, 
        llm=llm,
        cfg=cfg,
        output_path=output_path, 
        batch_size=batch_size,
        savename=savename, 
        evaluator_fn=evaluate, 
    )
