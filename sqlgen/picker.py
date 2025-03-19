### mubtasimahasan
from pathlib import Path
from typing import Callable
import pandas as pd
from vllm import LLM, SamplingParams
from core.base_agent import TextToSQL
from core.dbhandler import SQLiteDatabase


class CoderAgent(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame) -> tuple:
        return super().process_bird_df(idx, row)
    
    def generate_prompt(self, schema: str, question: str) -> str:
        return (
            "You are an expert SQL coder.\n\n"
            "Given the following SQLite tables, your job is to write a query to answer the given question.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}\n\n"
            f"### YOUR RESPONSE\nLet's think step by step"
        )


class VerdictReasoner(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, coder_outputs: list[str]) -> tuple:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, coder_outputs[idx]

    def generate_prompt(self, schema: str, question: str, coder_outputs: list[str]) -> str:
        responses = ''.join(
            f"###### Coder_{i+1}\n{resp}\n\n"
            for i, resp in enumerate(coder_outputs)
        )
        return (
            "You are a highly experienced SQL expert tasked with evaluating multiple SQL query responses from different coders.\n\n"
            "### Instructions:\n"
            "1. Carefully analyze the provided database schema and the given question.\n"
            "2. Review the SQL queries written by different coders. Consider the following criteria:\n"
            "   - **Correctness**: Does the query correctly retrieve the required data?\n"
            "   - **Efficiency**: Does the query minimize unnecessary computations?\n"
            "   - **Readability**: Is the SQL query well-structured and easy to understand?\n\n"
            "### SCHEMA\n"
            f"{schema}\n\n"
            "### QUESTION\n"
            f"{question}\n\n"
            "### CODER RESPONSES\n"
            f"{responses}"
            "### BEST RESPONSE\n"
            "Select the most optimal SQL query based on the criteria above. "
            "Provide reasoning for your choice before presenting the final SQL query.\n\n"
            "Let's think step by step.\n\n"
        )


class ReasonerPicker:
    @staticmethod
    def run(
        df: pd.DataFrame, databases: dict[str, SQLiteDatabase], 
        rzn_llm: LLM, code_llm: LLM, rzn_cfg: SamplingParams, code_cfg: SamplingParams,
        output_path: Path, batch_size: int, savename: str, evaluator_fn: Callable
    ) -> pd.DataFrame:

        coder_1 = CoderAgent(code_llm, databases, output_path)
        coder_2 = CoderAgent(code_llm, databases, output_path)
        coder_3 = CoderAgent(code_llm, databases, output_path)
        reasoner = VerdictReasoner(rzn_llm, databases, output_path)

        coder_1_outputs, _ = coder_1.batched_generate(df, code_cfg, batch_size, 'coder1', None)
        coder_2_outputs, _ = coder_2.batched_generate(df, code_cfg, batch_size, 'coder2', None)
        coder_3_outputs, _ = coder_3.batched_generate(df, code_cfg, batch_size, 'coder3', None)

        def collect_coder_responses():
            return [
                [coder_1_outputs.raw_responses[i], coder_2_outputs.raw_responses[i], coder_3_outputs.raw_responses[i]]
                for i in range(len(df))
            ]

        # print("\n========== CODER OUTPUTS ==========")
        # for i, coder_responses in enumerate(collect_coder_responses()):
        #     print(f"\nRow {i + 1}:")
        #     for j, response in enumerate(coder_responses):
        #         print(f"Coder {j + 1} Output:\n{response}\n")

        best_outputs, best_labels = reasoner.batched_generate(
            df, rzn_cfg, batch_size, 'reasoner', evaluator_fn, coder_outputs=collect_coder_responses()
        )

        # print("\n========== REASONER DECISIONS ==========")
        # for i, response in enumerate(best_outputs.raw_responses):
        #     print(f"\nRow {i + 1} Best Query:\n{response}\n")

        final_df = pd.concat([
            df,
            coder_1_outputs.as_dataframe(col_suffix='coder1'),
            coder_2_outputs.as_dataframe(col_suffix='coder2'),
            coder_3_outputs.as_dataframe(col_suffix='coder3'),
            best_outputs.as_dataframe(col_suffix='reasoner'),
            pd.DataFrame({'label_reasoner': best_labels}),
        ], axis=1)

        final_df.to_json(output_path / f"df_{savename}_final.json", orient='records')
        print('Finished MultiCoderReasoner experiment')
        return final_df