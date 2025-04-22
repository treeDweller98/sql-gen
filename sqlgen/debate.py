from typing import Callable
from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams
from core.base_agent import TextToSQL, TextToSQLGenerationOutput
from core.dbhandler import SQLiteDatabase
from sqlgen.zeroshot import ZeroShotAgent


class ProponentAgent(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, agent_responses: list[dict[int, str]]) -> tuple[str, str, dict[int, str]]:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, agent_responses[idx]
    
    def generate_prompt(self, schema: str, question: str, agent_responses: dict[int, str]) -> str:
        other_responses = ''.join(
            f"###### Agent_{agent}\n{response}\n\n"
            for agent, response in agent_responses.items()
        )
        prompt = (
            "You have been assigned the role of the Proponent. Given the following SQLite tables and the user question, "
            "your task is to argue in favor of the given SQLite query"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}\n\n"
            f"### AGENT RESPONSES\n{other_responses}"
            f"### YOUR RESPONSE\nLet's think step by step\n\n"
        )
        return prompt
    

class OpponentAgent(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, agent_responses: list[dict[int, str]]) -> tuple[str, str, dict[int, str]]:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, agent_responses[idx]
    
    def generate_prompt(self, schema: str, question: str, agent_responses: dict[int, str]) -> str:
        other_responses = ''.join(
            f"###### Agent_{agent}\n{response}\n\n"
            for agent, response in agent_responses.items()
        )
        prompt = (
            "You have been assigned the role of the Opponent. "
            f"### QUESTION\n{question}\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}\n\n"
            f"### AGENT RESPONSES\n{other_responses}"
            f"### YOUR RESPONSE\nLet's think step by step\n\n"
        )
        return prompt


class JudgeAgent(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, agent_responses: list[dict[int, str]]) -> tuple[str, str, dict[int, str]]:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, agent_responses[idx]
    
    def generate_prompt(self, schema: str, question: str, agent_responses: dict[int, str]) -> str:
        other_responses = ''.join(
            f"###### Agent_{agent}\n{response}\n\n"
            for agent, response in agent_responses.items()
        )
        prompt = (
            "You have been assigned the role of the Judge. "
            f"### QUESTION\n{question}\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}\n\n"
            f"### AGENT RESPONSES\n{other_responses}"
            f"### YOUR RESPONSE\nLet's think step by step\n\n"
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

        starter, starter_label = ZeroShotAgent(llm, databases).batched_generate(df, cfg, batch_size, output_path, savename, evaluator_fn)

