from pathlib import Path
from vllm import LLM, SamplingParams
from core.dbhandler import SQLiteDatabase
from sqlgen.base_agent import TextToSQL


class ReasonerZeroShot(TextToSQL):
    """ Zero-shot SQL generator utilizing a reasoning llm"""            
    def generate_prompt(self, schema: str, question: str) -> str:
        prompt = (
            "Given the following SQLite tables, your job is to write a query to answer the user question. "
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}.\n\n"
            f"### SQLite Query\n\n"
        )
        return prompt
    

class PlannerAgent(TextToSQL):
    def generate_prompt(self, schema: str, question: str):
        prompt = (
            "You are Query"
        )
        return prompt
    

class CoderAgent(TextToSQL):
    def generate_prompt(self, schema: str, question: str) -> str:
        prompt = (
            ""
        )
        return prompt