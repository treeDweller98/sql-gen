from datetime import datetime

import pandas as pd

from core.dbhandler import SQLiteDatabase
from core.metrics.eval_acc import calculate_accuracy
from core.metrics.eval_f1 import calculate_soft_f1
from core.metrics.eval_ves import calculate_rves


def evaluate(
    df: pd.DataFrame, databases: dict[str, SQLiteDatabase], pred_col: str, true_col: str = "gold_sql"
) -> tuple[list[bool], str]:
    print(f"\n--- Evaluating Performance | TrueCol: {true_col} | PredCol: {pred_col} ---")
    start_time = datetime.now()

    labels, ex_report = calculate_accuracy(df, databases, pred_col, true_col)

    print(ex_report)
    print(
        f"--- Evaluation Completed in {datetime.now() - start_time} | TrueCol: {true_col} | PredCol: {pred_col} ---\n"
    )
    return labels, str(ex_report)


def evaluate_all_metrics(
    df: pd.DataFrame,
    databases: dict[str, SQLiteDatabase],
    pred_col: str,
    true_col: str,
    iterate_num: int,
    meta_timeout: float,
    num_cpus: int,
):
    start_time = datetime.now()

    _, ex_report = calculate_accuracy(df, databases, pred_col, true_col, meta_timeout, num_cpus, suppress_prints=True)
    print(f"EX calculated (elapsed {datetime.now() - start_time})\n{ex_report}\n")

    f1_report = calculate_soft_f1(df, databases, pred_col, true_col, meta_timeout, num_cpus)
    print(f"Soft-F1 calculated (elapsed {datetime.now() - start_time})\n{f1_report}\n")

    rves_report = calculate_rves(df, databases, pred_col, true_col, iterate_num, meta_timeout, num_cpus)
    print(f"R-VES calculated (elapsed {datetime.now() - start_time})\n{rves_report}")

    report = {
        "ex_report": ex_report,
        "f1_report": f1_report,
        "rves_report": rves_report,
    }
    return report
