import multiprocessing as mp
import sys

import pandas as pd
from func_timeout import FunctionTimedOut, func_timeout
from tqdm import tqdm

from core.dbhandler import SQLiteDatabase


def _check_sql_correctness(args):
    """Worker to execute predicted and true SQL and check equality."""
    i, db_path, pred_sql, true_sql, question_id, meta_timeout, suppress_prints = args

    import sqlite3

    def execute_sql(sql, meta_timeout):
        def _inner():
            with sqlite3.connect(db_path, uri=True) as conn:
                rows = conn.execute(sql).fetchall()
            return rows

        try:
            return func_timeout(meta_timeout, _inner)
        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            return [("__timeout__",)]
        except Exception:
            return [("__error__",)]

    try:
        pred_res = execute_sql(pred_sql, meta_timeout)
        true_res = execute_sql(true_sql, meta_timeout)
    except Exception as e:
        if not suppress_prints:
            print(f"Q_{question_id}: {e.__class__.__name__} {e}")
        return (i, False)

    # Mark as incorrect if either timed out or errored
    if ("__timeout__",) in pred_res or ("__error__",) in pred_res:
        return (i, False)
    if ("__timeout__",) in true_res or ("__error__",) in true_res:
        return (i, False)

    label = set(pred_res) == set(true_res)
    return (i, label)


def get_correctness_labels(
    df: pd.DataFrame,
    databases: dict[str, SQLiteDatabase],
    pred_col: str,
    true_col: str,
    meta_timeout: float = 30.0,
    num_cpus: int = 4,
    suppress_prints: bool = False,
) -> list[bool]:
    """
    Parallelized EX (execution accuracy) correctness checker.

    Runs gold and predicted SQL queries on databases in parallel.
    Returns labels[i] = True if prediction matches ground truth results.
    """
    # Prepare jobs
    jobs = [
        (
            i,
            databases[row["db_id"]].db_path,
            row[pred_col],
            row[true_col],
            row.get("question_id", i),
            meta_timeout,
            suppress_prints,
        )
        for i, row in df.iterrows()
    ]

    # Parallel execution
    with mp.Pool(processes=num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_check_sql_correctness, jobs),
                total=len(jobs),
                desc="Executing SQL for EX",
            )
        )

    # Sort results back into original order
    results.sort(key=lambda x: x[0])
    labels = [r[1] for r in results]

    return labels


def calculate_accuracy(
    df: pd.DataFrame,
    databases: dict[str, SQLiteDatabase],
    pred_col: str,
    true_col: str,
    meta_timeout: float = 30.0,
    num_cpus: int = 4,
    suppress_prints: bool = False,
) -> tuple[list[bool], dict[str, str | float | dict[str, dict[str, float]]]]:
    labels = get_correctness_labels(
        df,
        databases,
        pred_col,
        true_col,
        meta_timeout,
        num_cpus,
        suppress_prints,
    )

    ex_report: dict[str, str | float | dict[str, dict[str, float]]] = {
        "true_col": true_col,
        "pred_col": pred_col,
        "overall_accuracy": (sum(labels) / len(labels)) * 100,
        "breakdown_by_difficulty": {},
    }

    for difficulty in df["difficulty"].unique():
        difficulty_mask = df["difficulty"] == difficulty
        correct_rows = [label for label, mask in zip(labels, difficulty_mask) if mask]
        n_correct = sum(correct_rows)
        n_total = sum(difficulty_mask)
        sub_accuracy = (n_correct / n_total) * 100 if n_total > 0 else 0.0

        ex_report["breakdown_by_difficulty"][difficulty] = {
            "accuracy": sub_accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
        }

    return labels, ex_report
