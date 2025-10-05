import multiprocessing as mp
import sys

import pandas as pd
from func_timeout import FunctionTimedOut, func_timeout
from tqdm import tqdm

from core.dbhandler import SQLiteDatabase


def _calculate_row_match(predicted_row, ground_truth_row):
    """Row-level overlap for Soft-F1."""
    total_columns = len(ground_truth_row)
    matches = sum(1 for val in predicted_row if val in ground_truth_row)
    pred_only = sum(1 for val in predicted_row if val not in ground_truth_row)
    truth_only = sum(1 for val in ground_truth_row if val not in predicted_row)

    return (
        matches / total_columns,
        pred_only / total_columns,
        truth_only / total_columns,
    )


def _calculate_f1_score(predicted, ground_truth):
    """Soft-F1 based on row overlaps."""
    # if both predicted and ground_truth are empty, return 1.0 for f1_score
    if not predicted and not ground_truth:
        return 1.0

    # Drop duplicates and convert back to list
    predicted = list(dict.fromkeys(predicted or []))
    ground_truth = list(dict.fromkeys(ground_truth))

    # Calculate matching scores for each possible pair
    match_scores, pred_only_scores, truth_only_scores = [], [], []
    for i, gt_row in enumerate(ground_truth):
        # rows only in the ground truth results
        if i >= len(predicted):
            match_scores.append(0)
            truth_only_scores.append(1)
            continue
        pred_row = predicted[i]
        m, p, t = _calculate_row_match(pred_row, gt_row)
        match_scores.append(m)
        pred_only_scores.append(p)
        truth_only_scores.append(t)

    # rows only in the predicted results
    for _ in range(len(predicted) - len(ground_truth)):
        match_scores.append(0)
        pred_only_scores.append(1)
        truth_only_scores.append(0)

    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1_score


def _process_row_for_f1(args):
    """Worker function to compute Soft-F1 for a single SQL pair."""
    idx, db_path, pred_sql, true_sql, timeout = args

    import sqlite3

    def execute_sql(sql, timeout):
        def _inner():
            with sqlite3.connect(db_path, uri=True) as conn:
                rows = conn.execute(sql).fetchall()
            return rows

        try:
            return func_timeout(timeout, _inner)
        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            return [("__timeout__",)]
        except Exception:
            return [("__error__",)]

    try:
        pred_rows = execute_sql(pred_sql, timeout)
        true_rows = execute_sql(true_sql, timeout)
    except Exception:
        return {"sql_idx": idx, "f1": 0.0}

    f1 = _calculate_f1_score(pred_rows, true_rows)
    return {"sql_idx": idx, "f1": f1}


def calculate_soft_f1(
    df: pd.DataFrame,
    databases: dict[str, SQLiteDatabase],
    pred_col: str,
    true_col: str,
    meta_timeout: float = 30.0,
    num_cpus: int = 4,
):
    """Multiprocessing-based Soft-F1 evaluation."""
    jobs = []
    for i, row in df.iterrows():
        db: SQLiteDatabase = databases[row["db_id"]]
        jobs.append((i, db.db_path, row[pred_col], row[true_col], meta_timeout))

    with mp.Pool(processes=num_cpus) as pool:
        exec_results = list(
            tqdm(
                pool.imap_unordered(_process_row_for_f1, jobs),
                total=len(jobs),
                desc="Calculating Soft-F1",
            )
        )

    exec_results = sorted(exec_results, key=lambda r: r["sql_idx"])

    simple, moderate, challenging = [], [], []
    for i, row in df.iterrows():
        r = exec_results[i]
        diff = row["difficulty"].lower()
        if diff == "simple":
            simple.append(r)
        elif diff == "moderate":
            moderate.append(r)
        elif diff == "challenging":
            challenging.append(r)

    def compute_mean(results: list[dict[str, float]]) -> float:
        return (sum(r["f1"] for r in results) / len(results) * 100) if results else 0.0

    report = {
        "true_col": true_col,
        "pred_col": pred_col,
        "soft_f1": compute_mean(exec_results),
        "breakdown_by_difficulty": {
            "simple_f1": compute_mean(simple),
            "moderate_f1": compute_mean(moderate),
            "challenging_f1": compute_mean(challenging),
        },
        "counts": [len(simple), len(moderate), len(challenging), len(exec_results)],
    }

    return report
