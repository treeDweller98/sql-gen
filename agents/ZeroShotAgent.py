import pandas as pd
from tqdm import tqdm

from core.Model import GenerationConfig
from agents.TextToSQL import TextToSQL


class ZeroShotAgent(TextToSQL):
    """ Zero-shot SQL Generator based on OpenAI Cookbook's "Natural Language to SQL" example and zero-shot COT. """
    def generate_response(self, schema: str, question: str, cfg: GenerationConfig) -> str:
        system_prompt = (
            f"Given the following SQLite tables, your job is to write queries given a user’s request. "
            f"Please begin your response with 'Let\'s think step by step'.\n\n{schema}"
        )
        user_prompt   = question
        reply = self.llm(
            messages=[
                { "role": "system", "content": system_prompt }, 
                { "role": "user",   "content": user_prompt },
            ],
            cfg=cfg
        )
        return reply
    
    def batched_generate(self, df: pd.DataFrame, cfg: GenerationConfig, savename: str = None) -> list[str]:
        assert {'db_id', 'question', 'evidence'}.issubset(df.columns), \
        "Ensure {'db_id', 'question', 'evidence'} in df.columns"

        raw_responses: list[str] = []
        for i, row in tqdm(df.iterrows(), desc=f'{self.agent_name} Generating SQL', total=len(df)):
            db = self.databases[ row['db_id'] ]
            schema = str(db)
            question = f"{row['question']}  Hint: {row['evidence']}"
            try:
                reply = self.generate_response(schema, question, cfg)
                raw_responses.append(reply)
            except Exception as e:
                self.dump_to_json_on_error(raw_responses)
                raise e
            
        if savename:
            self.dump_to_json(f"{savename}_raw", raw_responses)
            
        return raw_responses
    

class MetaPromptZeroShotAgent(ZeroShotAgent):
    """ Zero-shot SQL Generator using detailed meta-prompt of instructions"""
    def generate_response(self, schema: str, question: str, cfg: GenerationConfig) -> str:
        system_prompt = (
            "You are an SQLite expert who excels at writing queries. Your job is to write  "
            "a valid SQLite query to answer a given user question based on the schema below. "
            "Here is how you should approach the problem:\n"
            "1. Begin your response with 'Let\'s think step by step.'\n"
            "2. Analyze the question and schema carefully, showing all your workings:\n"
            "   - Decompose the question into subproblems.\n"
            "   - Identify the tables and the columns required to write the query.\n"
            "   - Identify the operations you will need to perform.\n"
            "3. Review your choices before generation:\n"
            "   - Identify if you missed any tables and columns.\n"
            "   - Identify if you picked any unnecessary tables and columns.\n"
            "   - Identify any unnecessary subqueries, joins, groupings, orderings, sortings etc.\n"
            "4. Ensure your choices are correct and optimal.\n"
            "5. Finally, show your reasoning and write down the SQL query.\n\n"
            f"### Schema:\n{schema}"
        )
        user_prompt = (
            f"### Question:\n{question}."
        )
        reply = self.llm(
            messages=[
                { "role": "system", "content": system_prompt }, 
                { "role": "user",   "content": user_prompt },
            ],
            cfg=cfg
        )
        return reply