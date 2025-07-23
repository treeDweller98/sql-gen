import pandas as pd
from core.base_agent import TextToSQL


class ReasonerPickerAgent(TextToSQL):
    def process_question(self, idx: int, row: pd.DataFrame, coder_outputs: pd.Series) -> tuple[str, str, str]:
        schema, question = super().process_question(idx, row)
        return schema, question, coder_outputs[idx]

    def generate_prompt(self, schema: str, question: str, coder_outputs: str) -> str:
        prompt = (
            "You are an expert database engineer. Based on the following SQLite tables, your job is to generate "
            "a single SQLite query to answer the given question. Junior members of your team have written you several "
            "candidate queries along with some of their reasonings. Consider their work when writing your answer.\n\n" 
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}\n\n"
            f"### CANDIDATE QUERIES\n{coder_outputs}\n\n"
            f"### YOUR RESPONSE"
        )
        return prompt