import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from core.birdeval import evaluate_all_metrics
from core.dbhandler import SQLiteDatabase
from utils import SupportedModels, read_bird_dataset, read_spider_dataset

DATASET = ["bird", "spider"][1]
EXPERIMENT = ["zs", "rzs", "pick", "plan-exec"][3]
RESULT_DIR = Path(f"results/results_{DATASET}/{EXPERIMENT}/")
RES_FILE = Path(f"results/{DATASET}_{EXPERIMENT}_REPORT.json")

ITERATE_NUM = 100
META_TIMEOUT = 30.0
NUM_CPUS = 36


def get_databases(dataset: str):
    if dataset == "spider":
        input_dir = Path("data/spider_data")
        db_foldername = "database"
        question_filename = "dev.json"
        read_dataset = read_spider_dataset
    else:
        input_dir = Path("data/bird_minidev")
        db_foldername = "dev_databases"
        question_filename = "mini_dev_sqlite.json"
        read_dataset = read_bird_dataset

    _, databases = read_dataset(
        input_dir,
        question_filename,
        db_foldername,
        None,
        30.0,
        False,
    )
    return databases


def run_eval(databases: dict[str, SQLiteDatabase], exp: str):
    all_results = {}
    for model in tqdm(SupportedModels, desc="Evaluating Results"):
        dir = RESULT_DIR / f"{model.name}_{EXPERIMENT}"
        file = dir / f"df_batgen_{exp}.json"

        if file.exists():
            print(f"\n\nStarting: {model.name}...")

            report = evaluate_all_metrics(
                pd.read_json(file),
                databases,
                f"parsed_sql_{exp}",
                "gold_sql",
                ITERATE_NUM,
                META_TIMEOUT,
                NUM_CPUS,
            )
            all_results[model.name] = report

            with open(dir / "report.json", "w") as f:
                json.dump(report, f, indent=4)

            print(f"Finished {model.name}.\n\n")
        else:
            if dir.exists():
                print(file, "DOES NOT EXIST")

    return all_results


def pick_exp_wrapper(databases: dict[str, SQLiteDatabase]):
    res = {}
    for exp in ["large_pick", "mid_pick", "small_pick"]:
        res[exp] = run_eval(databases, exp)
    return res


def plan_exp_wrapper(databases: dict[str, SQLiteDatabase]):
    res = {}
    exps = [
        "deepseek_r1_qwen_7b_plan_solo_plan-exec",
        "deepseek_r1_qwen_14b_plan_solo_plan-exec",
        "deepseek_r1_qwen_32b_plan_solo_plan-exec",
        "qwq_32b_plan_solo_plan-exec",
        "multi_plan-exec",
    ]
    for exp in exps:
        res[exp] = run_eval(databases, exp)
    return res


def mad_exp_wrapper(databases: dict[str, SQLiteDatabase]):
    res = {}
    exps = [""]
    for exp in exps:
        res[exp] = run_eval(databases, exp)
    return res


if __name__ == "__main__":
    databases = get_databases(DATASET)

    if EXPERIMENT == "pick":
        all_results = pick_exp_wrapper(databases)
    if EXPERIMENT == "mad":
        all_results = mad_exp_wrapper(databases)
    elif EXPERIMENT == "plan-exec":
        all_results = plan_exp_wrapper(databases)
    else:
        all_results = run_eval(databases, EXPERIMENT)

    with open(RES_FILE, "w") as f:
        json.dump(all_results, f, indent=4)
