from pathlib import Path
from vllm import LLM, SamplingParams
from core.dbhandler import SQLiteDatabase
from core.base_agent import TextToSQL


class ReasonerZeroShot(TextToSQL):
    """ Zero-shot SQL generator utilizing a reasoning llm"""            
    def generate_prompt(self, schema: str, question: str) -> str:
        prompt = (
            "Given the following SQLite tables, your job is to write a query to answer the user question. "
            f"### QUESTION\n{question}.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}.\n\n"
            f"### RESPONSE\n"
        )
        return prompt
    

class PlannerAgent(TextToSQL):
    def generate_prompt(self, schema: str, question: str):
        prompt = (
            "You are .."
        )
        return prompt
    

class CoderAgent(TextToSQL):
    def generate_prompt(self, schema: str, question: str) -> str:
        prompt = (
            ""
        )
        return prompt