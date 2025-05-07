from typing import Callable
from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams
from core.base_agent import TextToSQL, TextToSQLGenerationOutput
from core.dbhandler import SQLiteDatabase
from sqlgen.zeroshot import ZeroShotAgent


class ProponentAgent(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, solution: list[str], rebuttal: list[str]) -> tuple[str, str, str, str]:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, solution[idx], rebuttal[idx]
    
    def generate_prompt(self, schema: str, question: str, solution: str, rebuttal: str) -> str:
        prompt = (
            "You have been assigned the role of the Proponent. You are debating why your SQLite query "
            "correctly answers the question based on the given tables. Defend against your opponent's "
            "rebuttal and convince the judge that your solution is the best.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}\n\n"
            f"### YOUR SOLUTION\n{solution}\n\n"
            f"### REBUTTAL\n{rebuttal}\n\n"
            f"### YOUR RESPONSE"
        )
        return prompt
    

class OpponentAgent(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, solution: list[str], rebuttal: list[str]) -> tuple[str, str, str, str]:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, solution[idx], rebuttal[idx]
    
    def generate_prompt(self, schema: str, question: str, solution: str) -> str:
        prompt = (
            "You have been assigned the role of the Opponent. You are debating why the proposed SQLite query "
            "does not correctly answer the question based on the given tables. Show the flaws in the proposal "
            "and suggest a better query to correctly answers the question.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}\n\n"
            f"### PROPOSED SOLUTION\n{solution}"
            f"### YOUR RESPONSE"
        )
        return prompt


class JudgeAgent(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, agent_responses: list[dict[int, str]]) -> tuple[str, str, dict[int, str]]:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, agent_responses[idx]
    
    def generate_prompt(self, schema: str, question: str, affirmative: str, negative: str) -> str:
        prompt = (
            "You have been assigned the role of the Judge. Two agents are debating the correct SQLite query that correctly "
            "answers the question based on the given tables. IF you are convinced that a solution has been reached, "
            "output the final SQL query in your verdict. ELSE output 'CONTINUE' to continue the debate for another round.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}\n\n"
            f"### AGENT RESPONSES\n{other_responses}"
            f"### YOUR VERDICT"
        )
        return prompt
    

class MultiAgentDebate:
    def debate(
        df: pd.DataFrame, databases: dict[str, SQLiteDatabase], 
        llm: LLM, cfg: SamplingParams,
        output_path: Path, batch_size: int, savename: str, evaluator_fn: Callable
    ) -> pd.DataFrame:
        
        proponent = ProponentAgent(llm, databases)
        opponent = OpponentAgent(llm, databases)
        judge = JudgeAgent(llm, databases)
        
        starter, starter_label = ZeroShotAgent(llm, databases).batched_generate(df, cfg, batch_size, output_path, 'aff1', evaluator_fn)

        aff = starter['raw_response']

        for round in range(0,4):
            neg_rebuttal, reb_label = opponent.batched_generate(df, cfg, batch_size, output_path, "neg1", None, starter['aff1'])
            neg = neg_rebuttal['raw_response']
            judge_1 = judge.batched_generate(df, cfg, batch_size, output_path, "judge1", None, starter['aff1'])

        aff_rebuttal, aff_label = proponent.batched_generate(df, cfg, batch_size, output_path, f"{savename}_aff1", None)
