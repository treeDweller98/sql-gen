import sys
import time
import warnings
import subprocess
import json
from typing import Optional

import pandas as pd

from core.SQLiteDatabase import SQLiteDatabase
from core.Model import SupportedModels, GenerationConfig, LLM
from core.EvaluatorForBIRD import EvaluatorForBIRD

from api_keys import OPENAI_API_KEY
from config import (
    INPUT_PATH,
    OUTPUT_PATH,
    BIRD_QUESTION_FILENAME, 
    DATABASES_FOLDERNAME, 
    USE_FULL_DB, 
    USE_CACHED_SCHEMA,
    USE_DEBUG_DATASET,
    MODEL, 
    IS_INFERENCING_LOCALLY,
    MODEL_DEBUG_MODE, 
    EXPERIMENT,
    IS_PRINT_TO_FILE,
)

### BIRD Dataset Reader Function ###
def read_dataset() -> tuple[pd.DataFrame, list[str], dict[str, SQLiteDatabase]]:
    """ BIRD dataset reader function.
        1. Reads dataset into DataFrame from "INPUT_PATH/BIRD_QUESTION_FILENAME".
        2. Lists database names from folders in "INPUT_PATH/DB_FOLDERNAME/".
        3. If IS_USE_FULL_DB is False, returns debug subset of databases
           ['formula_1', 'debit_card_specializing', 'thrombosis_prediction'].
        4. Creates dict of SQLiteDatabases, indexed by db_name.
    """
    df = pd.read_json(INPUT_PATH / BIRD_QUESTION_FILENAME)

    if USE_FULL_DB:
        db_names: list[str] = [f.name for f in (INPUT_PATH / DATABASES_FOLDERNAME).iterdir()]
    else:
        db_names: list[str] = ['formula_1', 'debit_card_specializing', 'thrombosis_prediction']
        df = df[df['db_id'].isin(db_names)]

    databases: dict[str, SQLiteDatabase] = {
        db_id: SQLiteDatabase(db_id, (INPUT_PATH / DATABASES_FOLDERNAME), USE_CACHED_SCHEMA) 
        for db_id in db_names
    }

    if USE_DEBUG_DATASET:
        df = df.head()

    print(f'{db_names=}, {len(df)=}')
    return df, db_names, databases


### Deploy Ollama ###
def serve_ollama() -> None:
    """ Serves ollama if not already running and pulls model. """
    def is_ollama_up() -> bool:
        pname = 'ollama'
        try:
            call = subprocess.check_output(f"pidof {pname}", shell=True)
            return True
        except subprocess.CalledProcessError:
            return False
        
    if is_ollama_up():
        warnings.warn("Ollama already running. Proceeding without re-deployment...")
    else:
        process = subprocess.Popen(
            "ollama serve",
            shell=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        time.sleep(5)
        print(f"Ollama deployed (PID: {process.pid}).")


def pull_ollama_model(model: SupportedModels.Ollama) -> None:
    """ Use if you need to manually pull a model locally. """
    subprocess.check_call(f"ollama pull {model.value}", shell=True)


### Experiment Setup ###
def initialise_experiment():
    """ Initialises experiment according to settings in config.py"""
    # TODO: use a proper logger instead of hacking print()
    if IS_PRINT_TO_FILE:
        global temp_sys_stdout
        temp_sys_stdout = sys.stdout
        sys.stdout = open(OUTPUT_PATH / 'output.txt', 'wt')   # logs all prints in the output directory
    print(f"Experiment: {MODEL.value}_{EXPERIMENT}")

    # TODO: extend for use with multiple models at once
    if IS_INFERENCING_LOCALLY and MODEL in SupportedModels.Ollama:
        serve_ollama()

    # TODO: make use of api_key and base_url for cloud-hosted ollama
    llm = LLM(MODEL, is_debug=MODEL_DEBUG_MODE, api_key=OPENAI_API_KEY, base_url=None)

    # TODO: look up proper default values for Ollama and OpenAI
    if MODEL in SupportedModels.Ollama:
        cfg = GenerationConfig(temperature=0.7, top_p=1, max_tokens=2048, num_ctx=4096*8, seed=42)
    else:
        cfg = GenerationConfig(temperature=0.7, top_p=1, max_tokens=2048)

    df, db_names, databases = read_dataset()

    evaluator = EvaluatorForBIRD(databases)

    return df, db_names, databases, llm, cfg, evaluator


def cleanup_experiment(df: pd.DataFrame, results: str):
    df.to_json(OUTPUT_PATH / 'final_df.json', orient='records')
    with open( OUTPUT_PATH / 'results.txt', 'w') as f:
        f.write(results)

    print(f"Experiment Completed: {MODEL.value}_{EXPERIMENT}")
    print(results)
    if IS_PRINT_TO_FILE:
        sys.stdout = temp_sys_stdout


### General Utility functions ###
def dump_to_json(filename: str, objects: list) -> None:
    """ Dumps a list of objects to OUTPUT_PATH/filename.json; use for keeping backups. """
    filepath = OUTPUT_PATH / f"{filename}.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(objects, f, ensure_ascii=False, indent=4)