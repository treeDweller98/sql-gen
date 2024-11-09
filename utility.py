import json
import sqlite3
import pandas as pd
from openai import OpenAI
from config import INPUT_PATH, OUTPUT_PATH, MODEL, IS_USE_FULL_DB


### General Utility functions ###
def dump_to_json(filename: str, objects: list) -> None:
    """ Dumps a list to model_method/model_method_filename.json; used for keeping backups. """
    filepath = OUTPUT_PATH / f"{OUTPUT_PATH.stem}_{filename}.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(objects, f, ensure_ascii=False, indent=4)


### BIRD Dataset Reader Function ###
def read_dataset() -> tuple[pd.DataFrame, list[str]]:
    """ BIRD dataset reader function.
        1. Reads dataset into DataFrame from "INPUT_PATH/dev.json".
        2. Lists database names from folders in "INPUT_PATH/dev_databases/".
        3. If IS_USE_FULL_DB is False, returns debug subset of databases
           ['formula_1', 'debit_card_specializing', 'thrombosis_prediction'].
    """
    df = pd.read_json(INPUT_PATH / "dev.json")
    if IS_USE_FULL_DB:
        db_names: list[str] = [f.name for f in (INPUT_PATH / 'dev_databases').iterdir()]
    else:
        db_names: list[str] = ['formula_1', 'debit_card_specializing', 'thrombosis_prediction']
        df = df[df['db_id'].isin(db_names)]
    return df, db_names


### Database Utility Functions ###
def get_db_cursor(db_id: str) -> sqlite3.Cursor:
    """ Connects to db and returns a cursor. """
    db_path = (INPUT_PATH / 'dev_databases/' / db_id / db_id).with_suffix('.sqlite')
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    return cursor


def fetch_BIRD_schemas(db_names: list[str]) -> dict[str, str]:
    """ Returns a dictionary of BIRD db_schemas indexed by db_names. """
    
    def fetch_schema(db_id: str) -> str:
        """Returns the schema of all tables in a .sqlite database. """
        cursor = get_db_cursor(db_id)
        cursor = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schemas: list[str] = []
        for table in tables:
            table_name, = table
            if table_name != "sqlite_sequence":
                cursor = cursor.execute("SELECT sql FROM sqlite_master WHERE name=?;", [table_name])
                schema = cursor.fetchone()[0]
                schemas.append(schema)

        return "\n".join(schemas)

    return {db_id: fetch_schema(db_id) for db_id in db_names}


def get_openai_client() -> OpenAI:
    """ Returns an OpenAI client for use with ChatGPT or locally hosted Ollama. """
    if MODEL in {'gpt-4o', 'gpt-4o-mini'}:
        client = OpenAI()
    else:
        client = OpenAI(base_url = 'http://localhost:11434/v1', api_key='ollama')
    return client
