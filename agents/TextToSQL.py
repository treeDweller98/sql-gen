import re
import json
import sqlite3
from pathlib import Path
from abc import ABC
from collections.abc import Callable
from tqdm import tqdm
import pandas as pd
from core.SQLiteDatabase import SQLiteDatabase
from core.Model import LLM, GenerationConfig


class TextToSQL(ABC):
    """ Base class for all Text-to-SQL generation agents. """
    # TODO: maybe implement a prompt format cleanup function; 
    # [Does Prompt Formatting Have Any Impact on LLM Performance?](https://arxiv.org/pdf/2411.10541)

    def __init__(
        self, llm: LLM, databases: dict[str, SQLiteDatabase], 
        output_path: Path,
    ) -> None:
        """ Attributes
            ----------
                llm: LLM
                    Text generation model, hosted locally or on the cloud.
                databases: dict[str, SQLiteDatabase]
                    Dictionary of SQLiteDatabases indexed by db_id
                output_path: Path
                    Directory to dump output json
        """
        self.llm = llm
        self.databases = databases
        self.output_path = output_path
        self.agent_name = f"{self.llm}_{self.__class__.__name__}"

    def generate_response(self, schema: str, question: str, cfg: GenerationConfig) -> str:
        """ Takes a natural language question and db_schema, returns the generated raw response containing the final SQL. """
        raise NotImplementedError
    
    def batched_generate(self, df: pd.DataFrame, cfg: GenerationConfig, savename: str = None) -> list[str]:
        """ Runs generate_response() on a DataFrame of BIRD questions. 
            If savename is given, saves parsed SQL to savename_raw.json
        """
        raise NotImplementedError

    def auto_parse_sql(self, response: str) -> str:
        """ Extracts SQL from responses containing '''sql ... ''' using regex. 
            If regex search fails, attempts to parse using LLM.
            Returns cleaned SQL or an empty string.
        """
        def parse_with_regex(response: str) -> str:
            try:
                sql = re.search(r'```sql(.*?)```', response, re.DOTALL).group(1).strip()
            except AttributeError as e:
                sql = ''
            return sql

        matched = parse_with_regex(response)
        if not matched:
            print("Failed to parse with regex, attempting with LLM...")
            prompt = (
                "Please extract and enclose the SQL query from the text within "
                "a ```sql <<your response here>> ``` code block. Remove any additional"
                "text from your response, leaving only the SQL."
                f'### Text:\n{response}'
            )
            llm_parsed = self.llm(
                messages=[{'role': 'user', 'content': prompt}],
                cfg=GenerationConfig(temperature=0.4)
            )
            matched = parse_with_regex(llm_parsed)
            if matched:
                print("Successfully parsed with LLM.")
            else:
                print("Failed to parse with LLM. Returning empty string.")

        return matched

    def batched_parse_sql(self, raw_responses: list[str], savename: str = None) -> list[str]:
        """ Extracts SQL from LLM responses using auto_parse_sql(). 
            If savename is given, saves parsed SQL to savename_clean.json
        """
        parsed_sql = []
        for response in tqdm(raw_responses, desc=f'{self.agent_name} Parsing SQL', total=len(raw_responses)):
            try:
                sql = self.auto_parse_sql(response)    
                parsed_sql.append(sql)
            except Exception as e:
                self.dump_to_json_on_error(parsed_sql)
                raise e

        if savename:
            self.dump_to_json(f"{savename}_clean", parsed_sql)

        return parsed_sql
    
    def is_sql_same(self, db_id: str, query_1: str, query_2: str) -> bool:
        """ Executes SQL queries and returns True if outputs match, with no operation errors. """
        try:
            res_1 = self.databases[db_id].run_query(query_1)
            res_2 = self.databases[db_id].run_query(query_2)
        except sqlite3.OperationalError as e:
            print(f"{e.__class__.__name__} {e}")
            return False
        else:
            return set(res_1) == set(res_2)
        
    def dump_to_json(self, filename: str, obj: object) -> None:
        """ Dumps a list of objects to self.output_path/filename.json; use for keeping backups. """
        filepath = self.output_path / f"{filename}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)
        
    def dump_to_json_on_error(self, raw_responses: list[str]) -> None:
        """ Dumps raw responses into a json file; used in case of errors interrupting batched generation. """
        filename = f"{self.agent_name}_error_bak.json"
        self.dump_to_json(filename, raw_responses)
