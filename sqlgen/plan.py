from typing import Callable
import re
from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams
from core.dbhandler import SQLiteDatabase
from core.base_agent import TextToSQL


class PlannerAgent(TextToSQL):
    """ Uses a reasoning model to generate plans for constructing SQL queries. """
    def generate_prompt(self, schema: str, question: str) -> str:
        prompt = (
            "You are an expert database engineer. Your job is to analyze the given schema, and construct a "
            "step-by-step plan on how to answer the given question. The tables, columns and operations you "
            "outline will be used by students to generate final SQLite query.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}"
        )
        return prompt
    
    def parse_with_regex(self, response: str) -> str:
        """ Extracts answer by removing content in <think></think> using regex. """
        if "</think>" in response:
            response = "<think>" + response
        plan = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        return plan


class CoderAgent(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, plans: pd.Series, enable_zscot: bool = True) -> tuple:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, plans[idx], enable_zscot
    
    def generate_prompt(self, schema: str, question: str, plan: str, enable_zscot: bool) -> str:
        prompt = (
            f"Based on the given schema and plan, generate a single SQLite query to answer the question.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### PLAN\n{plan}\n\n"
            f"### QUESTION\n{question}\n\n"
            f"{'\n\nLet\'s think step by step' if enable_zscot else ''}"
        )
        return prompt
    
    
class MultiPlanCoderAgent(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, plans: pd.Series, enable_zscot: bool = True) -> tuple:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, plans[idx], enable_zscot
    
    def generate_prompt(self, schema: str, question: str, plan: str, enable_zscot: bool) -> str:
        prompt = (
            f"Based on the given schema and plans, generate a single SQLite query to answer the question.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### PLANS\n{plan}"
            f"### QUESTION\n{question}\n\n"
            f"{'\n\nLet\'s think step by step' if enable_zscot else ''}"
        )
        return prompt
    


class SinglePlannerCoding:
    def run_coder_on_plans(
        df: pd.DataFrame, databases: dict[str, SQLiteDatabase], 
        llm: LLM, cfg: SamplingParams, batch_size: int,
        output_path: Path, savename: str, evaluator_fn: Callable,
        plan_df: pd.DataFrame, enable_zscot: bool,
    ):
        agent_coder = CoderAgent(llm, databases)
        for model in plan_df.columns:
            plans = plan_df[model]
            output, labels = agent_coder.batched_generate(
                df, cfg, batch_size, output_path, f"{model}_{savename}", evaluator_fn, plans=plans, enable_zscot=enable_zscot,
            )


class MultiPlannerCoding:
    def run_coder_on_plans(
        df: pd.DataFrame, databases: dict[str, SQLiteDatabase], 
        llm: LLM, cfg: SamplingParams, batch_size: int,
        output_path: Path, savename: str, evaluator_fn: Callable,
        combined_plans: pd.Series, enable_zscot: bool,
    ):
        multiplan_agent_coder = MultiPlanCoderAgent(llm, databases)
        output, labels = multiplan_agent_coder.batched_generate(
            df, cfg, batch_size, output_path, savename, evaluator_fn, plans=combined_plans, enable_zscot=enable_zscot,
        )