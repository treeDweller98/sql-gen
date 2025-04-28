import re
import pandas as pd
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
    def process_bird_df(self, idx: int, row: pd.DataFrame, plan: str) -> tuple:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, plan
    
    def generate_prompt(self, schema: str, question: str, plan: str) -> str:
        prompt = (
            f"Based on the following plan and schema, generate an SQLite query to answer the question.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### PLAN\n{plan}\n\n"
            f"### QUESTION\n{question}\n\n"
            "Let's think step by step"
        )
        return prompt