import pandas as pd
from vllm import LLM, SamplingParams
from utils import load_llm, del_llm
from core.birdeval import evaluate
from core.dbhandler import SQLiteDatabase
from sqlgen.zeroshot import ZeroShotAgent, ReasonerZeroShot
from sqlgen.discussion import MultiAgentDiscussion
from sqlgen.debate import MultiAgentDebate
from sqlgen.plan import PlannerAgent, CoderAgent


def zeroshot_experiment(
    args, df: pd.DataFrame, databases: dict[str, SQLiteDatabase],
):
    llm = load_llm(args)
    cfg = SamplingParams(
        temperature=0,
        top_p=1,
        repetition_penalty=1.05,
        max_tokens=1024,
    )
    agent_zs = ZeroShotAgent(llm, databases)
    outputs, labels = agent_zs.batched_generate(
        df, cfg, args.BATCH_SIZE, args.OUTPUT_PATH, args.EXPERIMENT, evaluate
    )

def reasoner_zeroshot_experiment(
    args, df: pd.DataFrame, databases: dict[str, SQLiteDatabase],
):  
    llm = load_llm(args)
    cfg = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=30,
        repetition_penalty=1.0,
        max_tokens=4096*2,
    )
    agent_rzs = ReasonerZeroShot(llm, databases)
    outputs, labels = agent_rzs.batched_generate(
        df, cfg, args.BATCH_SIZE, args.OUTPUT_PATH, args.EXPERIMENT, evaluate
    )


def planner_plan_experiment(
    args, df: pd.DataFrame, databases: dict[str, SQLiteDatabase],
):  
    llm = load_llm(args)
    cfg = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=30,
        repetition_penalty=1.0,
        max_tokens=4096*4,
    )
    agent_plan = PlannerAgent(llm, databases)
    outputs, labels = agent_plan.batched_generate(
        df, cfg, args.BATCH_SIZE, args.OUTPUT_PATH, args.EXPERIMENT, None
    )

def planner_exec_experiment(
    args, df: pd.DataFrame, databases: dict[str, SQLiteDatabase],
):
    raise NotImplementedError
    # TODO: load dataframe of planner responses 
    llm = load_llm(args)
    cfg = SamplingParams(
        temperature=0,
        top_p=1,
        repetition_penalty=1.05,
        max_tokens=1024,
    )
    agent_coder = CoderAgent(llm, databases)
    outputs, labels = agent_coder.batched_generate(
        df, cfg, args.BATCH_SIZE, args.OUTPUT_PATH, args.EXPERIMENT, evaluate
    )


def discuss_experiment(
    args, df: pd.DataFrame, databases: dict[str, SQLiteDatabase],
):
    llm = load_llm(args)
    cfg = SamplingParams(
        temperature=0,
        top_p=1,
        repetition_penalty=1.05,
        max_tokens=1024*2,
    )
    MultiAgentDiscussion.discuss(
        df=df, 
        databases=databases, 
        llm=llm,
        cfg=cfg,
        batch_size=args.BATCH_SIZE,
        output_path=args.OUTPUT_PATH, 
        savename=args.EXPERIMENT, 
        evaluator_fn=evaluate, 
    )


def debate_experiment(
    args, df: pd.DataFrame, databases: dict[str, SQLiteDatabase],
):
    llm = load_llm(args)
    cfg = SamplingParams(
        temperature=0,
        top_p=1,
        repetition_penalty=1.05,
        max_tokens=1024*2,
    )
    MultiAgentDebate.debate(
        df=df, 
        databases=databases, 
        llm=llm,
        cfg=cfg,
        batch_size=args.BATCH_SIZE,
        output_path=args.OUTPUT_PATH, 
        savename=args.EXPERIMENT, 
        evaluator_fn=evaluate, 
    )

def reasoner_picker_experiment(
    args, df: pd.DataFrame, databases: dict[str, SQLiteDatabase],
):
    raise NotImplementedError