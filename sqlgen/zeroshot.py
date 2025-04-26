import pandas as pd
from core.base_agent import TextToSQL


# class ZeroShotAgent(TextToSQL):
#     """ Zero-shot SQL Generator based on OpenAI Cookbook's "Natural Language to SQL" example and zero-shot COT. """            
#     def generate_prompt(self, schema: str, question: str) -> str:
#         prompt = (
#             "Given the following SQLite tables, your job is to write a query to answer the given question.\n\n"
#             f"### SCHEMA\n{schema}\n\n"
#             f"### QUESTION\n{question}\n\n"
#             f"### YOUR RESPONSE\nLet's think step by step"
#         )
#         return prompt
    

# class ReasonerZeroShot(TextToSQL):
#     """ Zero-shot SQL generator utilizing a reasoning llm"""            
#     def generate_prompt(self, schema: str, question: str) -> str:
#         prompt = (
#             "Given the following SQLite tables, your job is to write a query to answer the given question.\n\n"
#             f"### SCHEMA\n{schema}\n\n"
#             f"### QUESTION\n{question}\n\n"
#             f"### YOUR RESPONSE"
#         )
#         return prompt


class ZeroShotAgent(TextToSQL):
    """ Zero-shot SQL Generator based on OpenAI Cookbook's "Natural Language to SQL" example and zero-shot COT. """            
    def generate_prompt(self, schema: str, question: str) -> str:
        prompt = (
            "Given the following SQL tables, your job is to write a SQLite query to answer the question.\n"
            f"{schema}\n"
            f"{question}\n"
            f"Let's think step by step"
        )
        return prompt
    

class ReasonerZeroShot(TextToSQL):
    """ Zero-shot SQL generator utilizing a reasoning llm"""            
    def generate_prompt(self, schema: str, question: str) -> str:
        prompt = (
            "Given the following SQL tables, your job is to write a SQLite query to answer the question.\n"
            f"{schema}\n"
            f"{question}"
        )
        return prompt