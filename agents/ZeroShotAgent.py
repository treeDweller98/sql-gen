import pandas as pd
from tqdm import tqdm
from agents.TextToSQL import TextToSQL


class ZeroShotAgent(TextToSQL):
    """ Zero-shot SQL Generator using the OpenAI Cookbook's "Natural Language to SQL" example. """
    def generate_response(self, schema: str, question: str, hint: str) -> str:
        system_prompt = f"Given the following SQLite tables, your job is to write queries given a user’s request.\n\n{schema}"
        user_prompt   = f"{question}  Hint: {hint}"
        reply = self.request_model(
            messages = [
                { "role": "system", "content": system_prompt }, 
                { "role": "user",   "content": user_prompt },
            ]
        )
        return reply
    
    def batched_generate(self, df: pd.DataFrame) -> list[str]:
        assert {'db_id', 'question', 'evidence'}.issubset(df.columns), \
        "Ensure {'db_id', 'question', 'evidence'} in df.columns"
        
        raw_responses: list[str] = []
        for i, row in tqdm(df.iterrows(), desc='Generating SQL', total=len(df)):
            schema   = self.db_schemas[row['db_id']]
            question = row['question']
            hint     = row['evidence']
            try:
                reply = self.generate_response(schema, question, hint)
                raw_responses.append(reply)
            except Exception as e:
                self.dump_to_json_on_error(raw_responses)
                raise e
        return raw_responses