from pathlib import Path
import json
import pandas as pd
from vllm import LLM, SamplingParams
from utils import load_llm, del_llm
from core.birdeval import evaluate
from core.dbhandler import SQLiteDatabase
from sqlgen.zeroshot import ZeroShotAgent, ReasonerZeroShot
from sqlgen.discussion import MultiAgentDiscussion
from sqlgen.debate import MultiAgentDebate
from sqlgen.plan import PlannerAgent, CoderAgent, MultiPlanCoderAgent, SinglePlannerCoding, MultiPlannerCoding


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
    def compile_plans() -> pd.DataFrame:
        base_dir = Path('results/rplan')
        data = {}
        for model_dir in base_dir.iterdir():
            plan_file = model_dir / 'df_batgen_plan.json'
            if plan_file.is_file():
                plan_df = pd.read_json(plan_file)
                plan_df.loc[plan_df['n_out_tokens_plan'] > 8192, 'parsed_sql_plan'] = (
                    "Plan corrupted. You will have to figure it out yourself"
                )
                data[model_dir.name] = plan_df['parsed_sql_plan'].to_list()
        plan_df = pd.DataFrame(data)
        return plan_df

    def combine_best_model_plans() -> pd.Series:
        selected_plans_df = plan_df[['deepseek_r1_qwen_32b_plan', 'qwq_32b_plan']]
        return selected_plans_df.apply(
            lambda row: "\n\n".join(f"# Plan {i+1}\n{row[col]}" for i, col in enumerate(selected_plans_df.columns)),
            axis=1
        )

    plan_df = compile_plans()                      # individual plans by each reasoning model
    combined_plans = combine_best_model_plans()    # plans by 32b R1 and QwQ models 
    llm = load_llm(args)

    if 'r1' in args.MODEL.value.lower() or 'qwq' in args.MODEL.value.lower():
        enable_zscot = False
        cfg = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=30,
            repetition_penalty=1.0,
            max_tokens=4096*2,
        )
    else:
        enable_zscot = True
        cfg = SamplingParams(
            temperature=0,
            top_p=1,
            repetition_penalty=1.05,
            max_tokens=1024,
        )

    SinglePlannerCoding.run_coder_on_plans(
        df, databases, llm, cfg, args.BATCH_SIZE, args.OUTPUT_PATH, f"solo_{args.EXPERIMENT}", evaluate, plan_df, enable_zscot,
    )
    MultiPlannerCoding.run_coder_on_plans(
        df, databases, llm, cfg, args.BATCH_SIZE, args.OUTPUT_PATH, f"multi_{args.EXPERIMENT}", evaluate, combined_plans, enable_zscot,
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