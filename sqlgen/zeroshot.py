import pandas as pd
from sqlgen.base_agent import TextToSQL

# TODO: long prompts performs poorly with small models


class ZeroShotAgent(TextToSQL):
    """ Zero-shot SQL Generator based on OpenAI Cookbook's "Natural Language to SQL" example and zero-shot COT. """            
    def generate_prompt(self, schema: str, question: str) -> str:
        prompt = (
            "Given the following SQLite tables, your job is to write queries given a user’s request. "
            f"### QUESTION\n{question}.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}.\n\n"
            f"### RESPONSE\nLet's think step by step "
        )
        return prompt
    


### DO NOT USE
class MetaPromptZeroShotAgent(ZeroShotAgent):
    """ Zero-shot SQL Generator using detailed meta-prompt of instructions"""
    def generate_prompt(self, schema: str, question: str) -> str:
        prompt = (
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
            f"### Schema:\n{schema}\n\n"
            f"### Question:\n{question}."
        )
        return prompt
        

class OptimizerAgent(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, pred_col=str) -> tuple:
        schema, question = super().process_bird_df(idx, row)
        sql = row[pred_col]
        return schema, question, sql

    def generate_prompt(self, schema: str, question: str, sql: str) -> str:
        prompt = (
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
            "3. Show your reasoning and write down the corrected, optimized, valid SQLite query.\n\n"
            f"### Schema:\n{schema}\n\n"
            f"### Question:\n{question}.\n\n"
            f"### SQL:\n{sql}"
        )
        return prompt