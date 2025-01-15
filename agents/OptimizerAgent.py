import pandas as pd
from tqdm import tqdm

from agents.TextToSQL import TextToSQL
from core.Model import GenerationConfig


class OptimizerAgent(TextToSQL):  
    def generate_response(self, schema: str, question: str, sql: str, cfg: GenerationConfig) -> str:
        system_prompt = (
            "You are an SQLite expert who excels at debugging and optimizing SQL queries. "
            "You will be given a database schema, a question, and an SQL query answering "
            "that question based on the given schema. Carefully analyse the schema, the "
            "question and the query. Your job is to do the following:\n"
            "1. Begin your response with 'Let\'s think step by step.'\n"
            "2. Analyze the query\n"
            "   - identify any invalid SQLite keywords.\n"
            "   - identify any invalid or missing columns and tables.\n"
            "   - identify any unnecessary subqueries, joins, groupings, orderings, sortings etc.\n"
            "   - ensure that query is a single SQL statement.\n"
            "3. Show your reasoning and write down the corrected, optimized, valid SQLite query."
        )
        user_prompt   = (
            f"### Schema:\n{schema}\n\n"
            f"### Question:\n{question}.\n\n"
            f"### SQL:\n{sql}"
        )
        reply = self.llm(
            messages=[
                { "role": "system", "content": system_prompt }, 
                { "role": "user",   "content": user_prompt },
            ],
            cfg=cfg
        )
        return reply
    
    def batched_generate(self, df: pd.DataFrame, pred_col: str, cfg: GenerationConfig, savename: str = None) -> list[str]:
        assert {'db_id', 'question', 'evidence', pred_col}.issubset(df.columns), \
        f"Ensure {'db_id', 'question', 'evidence', '{pred_col}'} in df.columns"
        
        raw_responses: list[str] = []
        for i, row in tqdm(df.iterrows(), desc=f'{self.agent_name} Generating SQL', total=len(df)):
            db = self.databases[ row['db_id'] ]
            schema = str(db)
            question = f"{row['question']}  Hint: {row['evidence']}"
            pred_sql = row[pred_col]
            try:
                reply = self.generate_response(schema, question, pred_sql, cfg)
                raw_responses.append(reply)
            except Exception as e:
                self.dump_to_json_on_error(raw_responses)
                raise e
            
        if savename:
            self.dump_to_json(f"{savename}_raw", raw_responses)

        return raw_responses