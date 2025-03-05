import pandas as pd
from pathlib import Path
from vllm import LLM, SamplingParams
from core.dbhandler import SQLiteDatabase
from sqlgen.base_agent import TextToSQL, TextToSQLGenerationOutput


class InitReasonerAgent(TextToSQL):
    def generate_prompt(self, schema: str, question: str) -> str:
        prompt = (
            "You are an expert SQL reasoner. Your job is to analyze the given schema and question, "
            "and provide a detailed reasoning for how to construct the SQL query. Your reasoning "
            "will be used by other agents to generate the actual SQL code.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}.\n\n"
            f"### REASONING\nLet's think step by step "
        )
        return prompt


class CoderAgent(TextToSQL):
    def generate_prompt(self, schema: str, question: str, reasoning: str) -> str:
        prompt = (
            f"Based on the following reasoning, generate an SQL query to answer the question.\n\n"
            f"### REASONING\n{reasoning}\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}.\n\n"
            f"### SQL QUERY\n"
        )
        return prompt


class ReasonerCoderTeam:
    def __init__(self, llm: LLM, databases: dict[str, SQLiteDatabase], output_path: Path):
        self.reasoner = InitReasonerAgent(llm, databases, output_path)
        self.coder_1 = CoderAgent(llm, databases, output_path)
        self.coder_2 = CoderAgent(llm, databases, output_path)
        self.coder_3 = CoderAgent(llm, databases, output_path)

    def generate_sql(self, df: pd.DataFrame, cfg: SamplingParams, batch_size: int, savename: str) -> pd.DataFrame:
        reasonings, _ = self.reasoner.batched_generate(df, cfg, batch_size, f'{savename}_reasoner')

        coder_1_outputs, _ = self.coder_1.batched_generate(df, cfg, batch_size, f'{savename}_coder1', reasoning=reasonings.raw_responses)
        coder_2_outputs, _ = self.coder_2.batched_generate(df, cfg, batch_size, f'{savename}_coder2', reasoning=reasonings.raw_responses)
        coder_3_outputs, _ = self.coder_3.batched_generate(df, cfg, batch_size, f'{savename}_coder3', reasoning=reasonings.raw_responses)

        final_df = pd.concat([
            df,
            reasonings.as_dataframe(col_suffix='reasoner'),
            coder_1_outputs.as_dataframe(col_suffix='coder1'),
            coder_2_outputs.as_dataframe(col_suffix='coder2'),
            coder_3_outputs.as_dataframe(col_suffix='coder3'),
        ], axis=1)

        final_df.to_json(self.reasoner.output_path / f"df_{savename}_final.json", orient='records')
        print('finished generating SQL')
        return final_df