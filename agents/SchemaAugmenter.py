import pandas as pd
from tqdm import tqdm

from agents.TextToSQL import TextToSQL
from core.Model import GenerationConfig


class SchemaAugmenterAgent(TextToSQL):  
    def generate_response(self, schema: str, question: str, sql: str, cfg: GenerationConfig) -> str:
        system_prompt = (
            "Please answer any requests from the user regarding the SQLite database given below."
            "Analyze the schema carefully. You will be asked regarding the purpose of each table "
            "and column, and you will be asked to explain the relationships between the tables.\n\n"
            f"### Schema:\n{schema}"
        )
        user_prompts = [
            "Can you please write a short description for each table?",

            "Can you explain the relationship between each table and summarise the information in a graph? "
            "The graph should be in {WHAT NOTATION TO USE?}",

            # What else to use this agent for?
            # Should we attempt to make this question-specific
        ]

        replies = [
            self.llm(
                messages=[
                    { "role": "system", "content": system_prompt }, 
                    { "role": "user",   "content": user_prompt },
                ],
                cfg=cfg
            )
            for user_prompt in user_prompts
        ]
        return replies
    
    def batched_generate(self, df: pd.DataFrame) -> list[str]:
        assert {'db_id'}.issubset(df.columns), \
        "Ensure {'db_id'} in df.columns"
        
        raw_responses: list[str] = []
        for i, row in tqdm(df.iterrows(), desc=f'{self.__agent_name} Augmenting Schemas', total=len(df)):
            schema   = self.db_schemas[row['db_id']]
            try:
                replies = self.generate_response(schema)
                raw_responses.append(replies)
            except Exception as e:
                self.dump_to_json_on_error(raw_responses)
                raise e
        return raw_responses