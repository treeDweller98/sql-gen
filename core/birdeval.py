### core/birdeval.py
import pandas as pd
from tqdm import tqdm
from core.dbhandler import SQLiteDatabase

def get_correctness_labels(df: pd.DataFrame, databases: dict[str, SQLiteDatabase], pred_col: str, true_col: str) -> list[bool]:
    ''' Takes DataFrame of BIRD questions with prediction column.
        Runs gold and predicted SQL queries on databases given.
        Returns labels, where labels[i] is True where pred_sql results same as ground_sql's
    '''
    labels = []
    for i, question in tqdm(df.iterrows(), desc='Executing SQL', total=len(df)):
        db = databases[question['db_id']]
        try:
            pred_res = db.run_query(question[pred_col])
            true_res = db.run_query(question[true_col])
        except Exception as e:
            print(f"Q_{question['question_id']}: {e.__class__.__name__} {e}")
            labels.append(False)
        else:
            labels.append( set(pred_res) == set(true_res) )
    return labels


def calculate_accuracy(df: pd.DataFrame, pred_col: str, true_col: str, labels: list[bool]) -> str:
    ex_report = (
        f"=== EX Results | TrueCol: {true_col} | PredCol: {pred_col} ===\n"
        f"Accuracy : {(sum(labels) / len(labels)) * 100: .3f}%\n"
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


# # TODO: add soft-f1 score to report
def calculate_softf1():
    raise NotImplementedError
def calculate_ves():
    raise NotImplementedError
def calculate_rves():
    raise NotImplementedError

    
def evaluate(df: pd.DataFrame, databases: dict[str, SQLiteDatabase], pred_col: str, true_col: str = 'gold_sql') -> tuple[list[bool], str]:
    print(f'\n--- Evaluating Performance | TrueCol: {true_col} | PredCol: {pred_col} ---')
    labels = get_correctness_labels(df, databases, pred_col, true_col)
    ex_report = calculate_accuracy(df, pred_col, true_col, labels)
    # f1_report = calculate_softf1(df, pred_col, true_col, labels)
    # ves_report = calculate_ves(df, pred_col, true_col, labels)
    # rves_report = calculate_rves(df, pred_col, true_col, labels)

    # report = "\n\n".join(ex_report, f1_report, ves_report, rves_report)
    report = ex_report      # until the rest gets implemented
    print(report)
    print(f'--- Evaluation Completed | TrueCol: {true_col} | PredCol: {pred_col} ---\n')
    return labels, report