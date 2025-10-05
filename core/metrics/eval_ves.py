import math
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd
from func_timeout import FunctionTimedOut
from tqdm import tqdm

from core.dbhandler import SQLiteDatabase


def _process_row_for_rves(args):
    """
    Worker function to compute reward for a single SQL pair.
    Runs inside a separate process.
    """
    idx, db_path, pred_sql, true_sql, iterate_num, meta_timeout = args

    # We import here to avoid pickling issues at module level
    import sqlite3
    import time

    from func_timeout import func_timeout

    def execute_sql_timed(sql, timeout):
        def _inner():
            with sqlite3.connect(db_path, uri=True) as conn:
                start = time.time()
                rows = conn.execute(sql).fetchall()
                elapsed = time.time() - start
            return rows, elapsed

        try:
            return func_timeout(timeout, _inner)
        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            return [("__timeout__",)], None
        except Exception:
            return [("__error__",)], None

    # First, check semantic equivalence
    pred_rows, _ = execute_sql_timed(pred_sql, meta_timeout)
    true_rows, _ = execute_sql_timed(true_sql, meta_timeout)

    if set(pred_rows) != set(true_rows):
        return {"sql_idx": idx, "reward": 0.0}

    # If equivalent, run multiple timed executions
    ratios = []
    for _ in range(iterate_num):
        pred_rows, pred_time = execute_sql_timed(pred_sql, meta_timeout)
        true_rows, true_time = execute_sql_timed(true_sql, meta_timeout)

        if pred_time is None or true_time is None or pred_time <= 0:
            return {"sql_idx": idx, "reward": 0.0}

        ratios.append(true_time / pred_time)

    def clean_abnormal(values):
        arr = np.asarray(values)
        mean, std = np.mean(arr), np.std(arr)
        return arr[(arr > mean - 3 * std) & (arr < mean + 3 * std)]

    processed = clean_abnormal(ratios)
    time_ratio = np.mean(processed) if processed.size > 0 else 0.0

    if time_ratio == 0:
        reward = 0.0
    elif time_ratio >= 2:
        reward = 1.25
    elif 1 <= time_ratio < 2:
        reward = 1.0
    elif 0.5 <= time_ratio < 1:
        reward = 0.75
    elif 0.25 <= time_ratio < 0.5:
        reward = 0.5
    else:
        reward = 0.25

    return {"sql_idx": idx, "reward": reward}


def calculate_rves(
    df: pd.DataFrame,
    databases: dict[str, SQLiteDatabase],
    pred_col: str,
    true_col: str,
    iterate_num: int = 100,
    meta_timeout: float = 30.0,
    num_cpus: int = 4,
):
    """
    Multiprocessing-based R-VES evaluation using SQLiteDatabase.run_query_timed.
    """
    jobs = []
    for i, row in df.iterrows():
        db: SQLiteDatabase = databases[row["db_id"]]
        jobs.append(
            (
                i,
                db.db_path,
                row[pred_col],
                row[true_col],
                iterate_num,
                meta_timeout,
            )
        )

    with mp.Pool(processes=num_cpus) as pool:
        exec_results = list(
            tqdm(pool.imap_unordered(_process_row_for_rves, jobs), total=len(jobs), desc="Calculating R-VES")
        )

    # Sort results by index for consistent ordering
    exec_results = sorted(exec_results, key=lambda r: r["sql_idx"])

    # Per-difficulty breakdown
    simple, moderate, challenging = [], [], []
    for i, row in df.iterrows():
        r = exec_results[i]
        if row["difficulty"].lower() == "simple":
            simple.append(r)
        elif row["difficulty"].lower() == "moderate":
            moderate.append(r)
        elif row["difficulty"].lower() == "challenging":
            challenging.append(r)

    def compute_rves(exec_results: list[dict[str, int | float]]) -> float:
        num_queries: int = len(exec_results)
        total_reward: float = 0.0
        count: int = 0
        for i, result in enumerate(exec_results):
            if result["reward"] != 0:
                count += 1
            total_reward += math.sqrt(result["reward"]) * 100
        ves: float = total_reward / num_queries
        return ves

    ves_report = {
        "true_col": true_col,
        "pred_col": pred_col,
        "rves": compute_rves(exec_results),
        "breakdown_by_difficulty": {
            "simple_rves": compute_rves(simple) if simple else 0.0,
            "moderate_rves": compute_rves(moderate) if moderate else 0.0,
            "challenging_rves": compute_rves(challenging) if challenging else 0.0,
        },
        "counts": [len(simple), len(moderate), len(challenging), len(exec_results)],
    }

    return ves_report
