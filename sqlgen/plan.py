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
            f"### QUESTION\n{question}\n\n"
            f"### YOUR RESPONSE"
        )
        return prompt
    
    def parse_with_regex(self, response: str) -> str:
        """ Extracts answer by removing content in <think></think> using regex. """
        if "</think>" in response:
            response = "<think>" + response
        plan = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        return plan


class CoderAgent(TextToSQL):
    def process_question(self, idx: int, row: pd.DataFrame, plans: pd.Series, enable_zscot: bool = True) -> tuple[str, str, str, bool]:
        schema, question = super().process_question(idx, row)
        return schema, question, plans[idx], enable_zscot
    
    def generate_prompt(self, schema: str, question: str, plan: str, enable_zscot: bool) -> str:
        cot = "\nLet's think step by step" if enable_zscot else '' 
        prompt = (
            f"Based on the given schema and plan, generate a single SQLite query to answer the question.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}\n\n"
            f"### PLAN\n{plan}\n\n"
            f"### YOUR RESPONSE"
            f"{cot}"
        )
        return prompt
    
    
class MultiPlanCoderAgent(TextToSQL):
    def process_question(self, idx: int, row: pd.DataFrame, plans: pd.Series, enable_zscot: bool = True) -> tuple[str, str, str, bool]:
        schema, question = super().process_question(idx, row)
        return schema, question, plans[idx], enable_zscot
    
    def generate_prompt(self, schema: str, question: str, plan: str, enable_zscot: bool) -> str:
        cot = "\nLet's think step by step" if enable_zscot else '' 
        prompt = (
            f"Based on the given schema and plans, generate a single SQLite query to answer the question.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}\n\n"
            f"### PLANS\n{plan}\n\n"
            f"### YOUR RESPONSE"
            f"{cot}"
        )
        return prompt