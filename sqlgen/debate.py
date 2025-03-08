from typing import Literal, Callable
from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams
from core.base_agent import TextToSQL, TextToSQLGenerationOutput
from core.dbhandler import SQLiteDatabase


class ZeroShotStarter(TextToSQL):
    """ Zero-shot SQL Generator based on OpenAI Cookbook's "Natural Language to SQL" example and zero-shot COT. """            
    def generate_prompt(self, schema: str, question: str) -> str:
        prompt = (
            "Given the following SQLite tables, your job is to write queries given a user’s request.\n\n"
            f"### QUESTION\n{question}.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}.\n\n"
            f"### RESPONSE\nLet's think step by step\n\n"
        )
        return prompt
    

class PositiveAgent(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame) -> tuple[str, str, dict[int, str]]:
        raise NotImplementedError
        schema, question = super().process_bird_df(idx, row)
        return schema, question
    
    def generate_prompt(self, schema: str, question: str, agent_responses: dict[int, str]) -> str:
        raise NotImplementedError
        other_responses = ''.join(
            f"###### Agent_{agent}\n{response}\n\n"
            for agent, response in agent_responses.items()
        )
        prompt = (
            f"### QUESTION\n{question}\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### AGENT RESPONSES\n{other_responses}"
            f"### QUESTION\n{question}\n\n"
            f"### YOUR RESPONSE\nLet's think step by step\n\n"
        )
        return prompt
    

class MultiAgentDebate:
    def debate(
        df: pd.DataFrame, databases: dict[str, SQLiteDatabase], 
        llm: LLM, cfg: SamplingParams,
        output_path: Path, batch_size: int, savename: str, evaluator_fn: Callable
    ) -> pd.DataFrame:
        raise NotImplementedError