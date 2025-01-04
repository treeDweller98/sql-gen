import time, sqlite3
import numpy as np
import pandas as pd
from collections.abc import Callable
from tqdm import tqdm

from core.SQLiteDatabase import SQLiteDatabase

# TODO: 
# - Efficiency score probably buggy
# - Add soft-F1 with accuracy
# - Make multi-threaded like original
# - Figure out what func_timeout does and add to implementation


class EvaluatorForBIRD:
    def __init__(self, databases: dict[str, SQLiteDatabase]):
        self.databases = databases
        
    def get_correctness_labels(self, df: pd.DataFrame, pred_col: str, true_col: str) -> list[bool]:
        ''' Takes DataFrame of BIRD questions with prediction column
            Runs gold and predicted SQL queries
            Returns labels, where labels[i] is True where pred_sql results same as ground_sql's
        '''
        labels = []
        for i, question in tqdm(df.iterrows(), desc='Executing SQL', total=len(df)):
            db = self.databases[question['db_id']]
            try:
                pred_res = db.run_query(question[pred_col])
                true_res = db.run_query(question[true_col])
            except Exception as e:
                print(f"Q_{question['question_id']}: {e.__class__.__name__} {e}")
                labels.append(False)
            else:
                labels.append( set(pred_res) == set(true_res) )
        return labels

    def calculate_accuracy(self, df: pd.DataFrame, pred_col: str, true_col: str, labels: list[bool] = None) -> str:
        labels = labels if labels else self.get_correctness_labels(df, pred_col, true_col)


        ex_report = (
            "=== EX Results ==="
            f"Accuracy : {(sum(labels) / len(labels)) * 100: .3f}%"
            "Breakdown by Difficulty:\n"
        )    
        for difficulty in df['difficulty'].unique():
            difficulty_mask = df['difficulty'] == difficulty
            correct_rows = [label for label, mask in zip(labels, difficulty_mask) if mask]
            n_correct = sum(correct_rows)
            n_total = sum(difficulty_mask)
            sub_accuracy = (n_correct / n_total) * 100
            ex_report += f"\t{difficulty}: {sub_accuracy: .3f}% ({n_correct} of {n_total})\n"
        ex_report += '=== end ===\n'

        return ex_report
    
    # TODO: add soft-f1 score to report
    def calculate_soft_f1():
        raise NotImplementedError
    
    # TODO: figure out why time ratios come out weird
    # def get_exec_time_ratios(self, correct_questions: pd.DataFrame, pred_col: str, true_col: str = 'SQL', repeat: int = 100) -> list[float]:
        
    #     def get_execution_time(sql: str, cursor: sqlite3.Cursor) -> float:
    #         start_time = time.time()
    #         cursor.execute(sql)
    #         exec_time = time.time() - start_time
    #         return exec_time
        
    #     def reject_outliers_and_average(arr: list[float], m: int = 3) -> float:
    #         ''' keep only arr values within a m-sigma range (not sure how it helps)'''
    #         data = np.array(arr)
    #         cleaned = data[ np.abs(data - np.mean(data)) < m*np.std(data) ]
    #         cleaned_mean = np.mean(cleaned)
    #         return float(cleaned_mean)
        
    #     time_ratios = []
    #     for i, question in tqdm(correct_questions.iterrows(), desc="Calculating SQL exec times", total=len(correct_questions)):
    #         cursor = self.get_db_cursor(question['db_id'])
    #         exec_time_ratios = []
    #         for i in range(repeat):
    #             true_sql_time = get_execution_time(question[true_col], cursor)
    #             pred_sql_time = get_execution_time(question[pred_col], cursor)
    #             exec_time_ratios.append(true_sql_time / pred_sql_time)
    #         mean_exec_time_ratio = reject_outliers_and_average(exec_time_ratios)
    #         time_ratios.append(mean_exec_time_ratio)

    #     return time_ratios

    # def calculate_efficiency(self, df: pd.DataFrame, pred_col: str, true_col: str = 'SQL', labels: list[bool] = None) -> None:

    #     labels = labels if labels else self.get_correctness_labels(df, pred_col)
    #     correct_questions = df[labels]
    #     time_ratios = np.array(self.get_exec_time_ratios(correct_questions, repeat = 100))
        
    #     def map_rewards_to_ratios(time_ratios: float) -> float:
    #         def reward(ratio: float) -> float:
    #             match ratio:
    #                 case ratio if (0.00 <  ratio <  0.25): return 0.25
    #                 case ratio if (0.25 <= ratio <  0.50): return 0.50
    #                 case ratio if (0.50 <= ratio <  1.00): return 0.75
    #                 case ratio if (1.00 <= ratio <  2.00): return 1.00 
    #                 case ratio if (ratio >= 2.00): return 1.25
    #                 case _: return 0.00

    #         rewards = np.array([reward(ratio) for ratio in time_ratios])
    #         return rewards
        
    #     def calculate_ves_score(time_ratios):
    #         total_ratio = np.sum(np.sqrt(time_ratios) * 100)
    #         efficiency_score = total_ratio / len(time_ratios)
    #         return efficiency_score
        
    #     ves_score  = calculate_ves_score(time_ratios)
    #     rves_score = calculate_ves_score(map_rewards_to_ratios(time_ratios))

    #     print('=== Efficiency Results ===')
    #     print(f"VES   : {ves_score: .3f}%")
    #     print(f"R-VES : {rves_score: .3f}%")
    #     print(f"Breakdown by Difficulty:")
    #     for difficulty in correct_questions['difficulty'].unique():
    #         masked_ratios = time_ratios[ correct_questions['difficulty'] == difficulty ]
    #         print(f"\tVES   {difficulty}: {calculate_ves_score(masked_ratios): .3f}%")
    #         print(f"\tR-VES {difficulty}: {calculate_ves_score(map_rewards_to_ratios(masked_ratios)): .3f}%")
    #     print('=== end ===\n')            

    
    def evaluate(self, df: pd.DataFrame, pred_col: str, true_col: str = 'SQL') -> list[bool]:
        print('--- Evaluating Performance ---')
        labels = self.get_correctness_labels(df, pred_col, true_col)
        ex_report = self.calculate_accuracy(df, pred_col, true_col, labels)
        #ves_report = self.calculate_efficiency(df, pred_col, labels)
        return labels, ex_report #, ves_report