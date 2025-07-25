from pathlib import Path
import json
import pandas as pd
from vllm import LLM, SamplingParams
from utils import load_llm, del_llm
from core.birdeval import evaluate
from core.dbhandler import SQLiteDatabase
from sqlgen.zeroshot import ZeroShotAgent, ReasonerZeroShot
from sqlgen.discussion import MultiAgentDiscussion
from sqlgen.picker import ReasonerPickerAgent
from sqlgen.plan import PlannerAgent, CoderAgent, MultiPlanCoderAgent


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
        df, cfg, args.BATCH_SIZE, args.OUTPUT_DIR, args.EXPERIMENT, evaluate
    )
    del_llm(llm)

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
        df, cfg, args.BATCH_SIZE, args.OUTPUT_DIR, args.EXPERIMENT, evaluate
    )
    del_llm(llm)


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
        df, cfg, args.BATCH_SIZE, args.OUTPUT_DIR, args.EXPERIMENT, None
    )
    del_llm(llm)


def planner_exec_experiment(
    args, df: pd.DataFrame, databases: dict[str, SQLiteDatabase],
):  
    def compile_plans() -> pd.DataFrame:
        base_dir = Path(f'results_{args.DATASET}/rplan')
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
            lambda row: "\n\n".join(f"###### Plan {i+1}\n{row[col].replace('### ', '')}" for i, col in enumerate(selected_plans_df.columns)),
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

    plan_exec_agent = CoderAgent(llm, databases)
    for model in plan_df.columns:
        _, _ = plan_exec_agent.batched_generate(
            df, cfg, args.BATCH_SIZE, args.OUTPUT_DIR, f"{model}_solo_{args.EXPERIMENT}",
            evaluate, plans=plan_df[model], enable_zscot=enable_zscot,
        )

    multiplan_exec_agent = MultiPlanCoderAgent(llm, databases)
    _, _ = multiplan_exec_agent.batched_generate(
        df, cfg, args.BATCH_SIZE, args.OUTPUT_DIR, f"multi_{args.EXPERIMENT}", 
        evaluate, plans=combined_plans, enable_zscot=enable_zscot,
    )
    del_llm(llm)


def reasoner_picker_experiment(
    args, df: pd.DataFrame, databases: dict[str, SQLiteDatabase],
):
    def compile_zeroshot_outputs() -> dict[str, pd.Series]:
        small_models = {}
        mid_models = {}
        large_models = {}
        
        matched_dirs = [
            d for d in Path(f'results_{args.DATASET}/zs').iterdir() 
            if d.is_dir() and any(d.name.startswith(prefix) 
            for prefix in ['gemma3', 'qwen25', 'qwen25_coder'])
        ]
        for model_dir in matched_dirs:
            dirname = str(model_dir.name)
            responses_file = model_dir / 'zs_raw.json'
            if responses_file.is_file():
                with responses_file.open('r') as f:
                    responses = json.load(f)         
        
                if "_4b_" in dirname or '_7b_' in dirname:
                    small_models[dirname] = responses
                elif "_14b_" in dirname or '_12b_' in dirname:
                    mid_models[dirname] = responses
                elif "_32b_" in dirname or '_27b_' in dirname:
                    large_models[dirname] = responses

        def combine_zs_responses(model_responses_dict) -> pd.Series:
            df = pd.DataFrame.from_dict(model_responses_dict)
            return df.apply(
                lambda row: "\n\n".join(f"###### Candidate {i+1}\n{row[col].replace('### ', '')}" for i, col in enumerate(df.columns)),
                axis=1
            )
        
        return {
            'small': combine_zs_responses(small_models),
            'mid':   combine_zs_responses(mid_models),
            'large': combine_zs_responses(large_models),
        }
        
    responses = compile_zeroshot_outputs()
    llm = load_llm(args)
    cfg = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=30,
        repetition_penalty=1.0,
        max_tokens=4096*2,
    )

    agent_pick = ReasonerPickerAgent(llm, databases)
    for size, coder_outputs in responses.items():
        _, _ = agent_pick.batched_generate(
            df, cfg, args.BATCH_SIZE, args.OUTPUT_DIR, f"{size}_{args.EXPERIMENT}", 
            evaluate, coder_outputs=coder_outputs
        )
    del_llm(llm)


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
        output_dir=args.OUTPUT_DIR, 
        savename=args.EXPERIMENT, 
        evaluator_fn=evaluate, 
    )
    del_llm(llm)