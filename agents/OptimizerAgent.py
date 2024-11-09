import pandas as pd
from tqdm import tqdm
from agents.TextToSQL import TextToSQL


class OptimizerAgent(TextToSQL):  
    def generate_response(self, schema: str, question: str, hint: str, sql: str) -> str:
        system_prompt = (
            "You are an SQLite expert who excels at debugging and optimizing SQL queries. "
            "You will be given a database schema, a question, and a SQL query answering "
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
            f"### Question:\n{question} Hint: {hint}.\n\n"
            f"### SQL:\n{sql}"
        )
        reply = self.request_model(
            messages = [
                { "role": "system", "content": system_prompt }, 
                { "role": "user",   "content": user_prompt },
            ]
        )
        return reply
    
    def batched_generate(self, df: pd.DataFrame) -> list[str]:
        assert {'db_id', 'question', 'evidence', 'prediction'}.issubset(df.columns), \
        "Ensure {'db_id', 'question', 'evidence', 'prediction'} in df.columns"
        
        raw_responses: list[str] = []
        for i, row in tqdm(df.iterrows(), desc='Generating SQL', total=len(df)):
            schema   = self.db_schemas[row['db_id']]
            question = row['question']
            hint     = row['evidence']
            pred_sql = row['predicted']
            try:
                reply = self.generate_response(schema, question, hint, pred_sql)
                raw_responses.append(reply)
            except Exception as e:
                self.dump_to_json_on_error(raw_responses)
                raise e
        return raw_responses