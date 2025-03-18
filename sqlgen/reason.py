from pathlib import Path
from vllm import LLM, SamplingParams
from core.dbhandler import SQLiteDatabase
from core.base_agent import TextToSQL
    

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