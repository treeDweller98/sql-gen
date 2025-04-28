import pandas as pd
from datasets import Dataset
from core.dbhandler import SQLiteDatabase


def make_bird_grpo_dataset(df: pd.DataFrame, databases: dict[str, SQLiteDatabase]) -> Dataset:

    def make_prompt(row) -> list[dict[str, str]]:
        """ Takes a row of a DataFrame of BIRD Bench questions.
            Creates a prompt for the question.
        """
        schema = str(databases[row.db_id])
        question = f"Question: {row.question}  Context: {row.evidence}"
        system_prompt = (
            "You are a helpful SQLite coding assistant. Please answer user queries in the following format:\n\n"
            "<think>\n"
            "USE THIS AS A SCRATCHPAD FOR YOUR REASONING\n"
            "</think>\n"
            "THE DETAILED STEP BY STEP REASONING TO DERIVE THE ANSWER\n"
            "```sql\n"
            "YOUR FINAL SQLITE QUERY\n"
            "```"
        )
        user_prompt = (
            f"Given the following SQLite tables, your job is to write a single query to answer the given question.\n\n"
            f"{schema}\n\n"
            f"{question}.\n\n"
        )
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
    
    df['prompt'] = df.apply(make_prompt, axis=1)
    dataset = Dataset.from_pandas(df)

    return dataset