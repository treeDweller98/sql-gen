import re
from typing import Sequence
from grpo.reward_utils import (
    parse_with_regex,
    is_sql_valid,
    is_sql_same,
    extract_table_column_names,
    extract_hallucinated_table_col,
    case_insensitive_diff,
)

def correctness_reward_func(completions, db_id: Sequence[str], gold_sql: Sequence[str], **kwargs) -> list[float]:
    """ Rewards 2.0 IF outputs of pred_sql and gold_sql match ELSE 0.0 """
    pred_sql = [ parse_with_regex(completion[0]['content']) for completion in completions ]
    return [
        float(is_sql_same(db, pred, true)) * 2
        for db, pred, true in zip(db_id, pred_sql, gold_sql)
    ]

def valid_sql_reward_func(completions, **kwargs) -> list[float]:
    """ Rewards 0.5 IF pred_sql executes without OperationalError ELSE 0.0 """
    pred_sql = [ parse_with_regex(completion[0]['content']) for completion in completions ]
    return [
        float(is_sql_valid(sql)) * 0.5
        for sql in pred_sql
    ]

def table_col_reward_func(completions, db_id: Sequence[str], gold_sql: Sequence[str], **kwargs) -> list[float]:
    """ Rewards 1.0 IF gold_sql and pred_sql have the same tables and columns (heuristically matched) """
    pred_sql = [ parse_with_regex(completion[0]['content']) for completion in completions ]
    pred_table_col: list[set[str]] = [extract_table_column_names(sql) for sql in pred_sql]
    gold_table_col: list[set[str]] = [extract_table_column_names(sql) for sql in gold_sql]
    
    missing:      list[set[str]] = [case_insensitive_diff(gold, pred) for pred, gold in zip(pred_table_col, gold_table_col)]
    unnecessary:  list[set[str]] = [case_insensitive_diff(pred, gold) for pred, gold in zip(pred_table_col, gold_table_col)]
    hallucinated: list[set[str]] = [extract_hallucinated_table_col(id, pred) for id, pred in zip(db_id, pred_table_col) ]

    rewards = []
    
    ### ChatGPT wrote this --- DO NOT TRUST
    for i in range(len(pred_table_col)):
        # If there are no missing or unnecessary columns, it's a perfect match
        if not missing[i] and not unnecessary[i]:
            rewards.append(1.0)
        else:
            # Calculate penalty for missing and unnecessary columns
            missing_penalty      = len(missing[i])      / len(gold_table_col[i]) if gold_table_col[i] else 0
            unnecessary_penalty  = len(unnecessary[i])  / len(pred_table_col[i]) if pred_table_col[i] else 0
            hallucinated_penalty = len(hallucinated[i]) / len(pred_table_col[i]) if pred_table_col[i] else 0
            
            # Penalize for hallucinated columns, missing, and unnecessary columns
            total_penalty = missing_penalty + unnecessary_penalty + hallucinated_penalty
            
            # The reward is 1.0 minus the total penalty
            reward = max(0.0, 1.0 - total_penalty)
            rewards.append(reward)
    ### ChatGPT wrote this --- DO NOT TRUST

    return rewards

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """ Rewards 0.5 IF response has think and sql blocks ELSE 0.0 """
    pattern = r"<think>.*?</think>\s.*?```sql\s.*?```"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """ Rewards 0.5 IF response starts with think, contains explanation, and finally sql  ELSE 0.0 """
    pattern = r"^<think>\n.*?\n</think>\n.*?\n```sql\n.*?\n```$"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def tags_count_reward_func(completions, **kwargs) -> list[float]:
    """ Rewards upto 0.5 IF think and sql blocks occur only once.
        Penalizes characters occurring after an sql block by 0.001.
    """
    def tag_reward(text: str) -> float:
        reward = 0.0
        if text.count("<think>") == 1:
            reward += 0.125
        if text.count("</think>") == 1:
            reward += 0.125
        if len(re.findall(r"```sql\s.*?```", text, flags=re.DOTALL)) == 1:
            reward += 0.125  # sql block present'
            trailing = text.split(re.search(r"```sql\s.*?```", text).group(0), 1)[-1]
            reward -= len(trailing)*0.001    # penalize every char after sql block
        if len(re.findall(r"\s```sql\s.*?```$", text, flags=re.DOTALL)) == 1:
            reward += 0.125  # sql block is distinct and at the very end
        return reward
    
    responses = [completion[0]['content'] for completion in completions]
    return [tag_reward(response) for response in responses]



if __name__ == '__main__':
    from pathlib import Path
    import pandas as pd
    from core.dbhandler import SQLiteDatabase
    from grpo.reward_utils import set_db

    input_path  = Path(f'data/bird-minidev')
    bird_question_filename = 'dev.json'
    db_foldername = 'dev_databases'
    db_exec_timeout = 30.0
    use_cached_schema = False  

    df = pd.read_json(input_path / bird_question_filename)
    databases: dict[str, SQLiteDatabase] = {
        db_id: SQLiteDatabase(db_id, (input_path / db_foldername), db_exec_timeout, use_cached_schema) 
        for db_id in [f.name for f in (input_path / db_foldername).iterdir()]
    }
    set_db(databases)
        
    # correct, wrong, wrong
    correct = "SELECT SUM(Consumption) AS TotalConsumption\nFROM yearmonth\nWHERE CustomerID = 6 AND Date BETWEEN '201308' AND '201311'",
    completions = [
        [{
            'db_id': 'debit_card_specializing',
            'gold_sql': "SELECT SUM(Consumption) FROM yearmonth WHERE CustomerID = 6 AND Date BETWEEN '201308' AND '201311'",
            'completions': f"<think>\nHere are some thoughts\n</think>\nHere is my summary and answer\n```sql\n{correct}\n```",
        }],
        [{
            'db_id': 'debit_card_specializing',
            'gold_sql': "SELECT CAST(SUM(IIF(Currency = 'EUR', 1, 0)) AS FLOAT) / SUM(IIF(Currency = 'CZK', 1, 0)) AS ratio FROM customers",
            'completions': "SELECT CAST(COUNT(CASE WHEN Currency = 'EUR' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(CASE WHEN Currency = 'CZK' THEN 1 ELSE NULL END) FROM customers", 
        }],
        [{
            'db_id': 'debit_card_specializing',
            'gold_sql': "SELECT T1.CustomerID FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = 'LAM' AND SUBSTR(T2.Date, 1, 4) = '2012' GROUP BY T1.CustomerID ORDER BY SUM(T2.Consumption) ASC LIMIT 1",
            'completions': "SELECT T1.CustomerID FROM customers AS T1 INNER JOIN gasstations AS T2 ON T1.CustomerID = T2.GasStationID INNER JOIN yearmonth AS T3 ON T1.CustomerID = T3.CustomerID WHERE SUBSTR(T3.Date, 1, 4) = '2012' GROUP BY T1.CustomerID ORDER BY SUM(T3.Consumption) ASC LIMIT 1",
        }],


    ]

    
