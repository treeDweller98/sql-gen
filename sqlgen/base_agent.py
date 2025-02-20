import re
import json
import sqlite3
from pathlib import Path
from abc import ABC
from collections.abc import Optional, Callable, Sequence
from tqdm import tqdm
import pandas as pd
from vllm import LLM, SamplingParams
from core.dbhandler import SQLiteDatabase


def batched(sequence: Sequence, n: int=1):
    # Same as `from itertools import batched`
    l = len(sequence)
    for ndx in range(0, l, n):
        yield sequence[ndx:min(ndx + n, l)]

        
class TextToSQLGenerationOutput:
    def __init__(
        self, input_prompts: list[str], raw_responses: list[str], parsed_sql: list[str], 
        n_in_tokens: list[int], n_out_tokens: list[int]
    ) -> None:
        self.input_prompts = input_prompts
        self.raw_responses = raw_responses
        self.parsed_sql = parsed_sql
        self.n_in_tokens = n_in_tokens
        self.n_out_tokens = n_out_tokens

    def as_dataframe(self, col_suffix: Optional[str] = '') -> pd.DataFrame:
        if col_suffix: 
            col_suffix = f'_{col_suffix}'
        df = pd.DataFrame({
            f'input_prompts{col_suffix}': self.input_prompts,
            f'raw_responses{col_suffix}': self.raw_responses,
            f'parsed_sql{col_suffix}':    self.parsed_sql,
            f'n_in_tokens{col_suffix}':   self.n_in_tokens,
            f'n_out_tokens{col_suffix}':  self.n_out_tokens,
        })
        return df

    def __str__(self):
        print("Input Prompts: ", self.input_prompts)
        print("Raw Responses: ", self.raw_responses)
        print("Parsed SQL: ", self.parsed_sql)
        print("Total Input Tokens = ", sum(self.n_in_tokens))
        print("Total Input Tokens = ", sum(self.n_out_tokens))
        
        
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
                    Text generation model for offline generation
                databases: dict[str, SQLiteDatabase]
                    Dictionary of SQLiteDatabases indexed by db_id
                output_path: Path
                    Directory to dump output json
        """
        self.llm = llm
        self.databases = databases
        self.output_path = output_path

    def process_bird_df(self, idx: int, row: pd.DataFrame, **kwargs) -> tuple:
        """ Takes a row of a DataFrame of BIRD Bench questions. 
            Processes and returns necessary columns required by this Agent's generate_response(). 
            Output tuple must be unpackable as parameters to generate_response().
        """
        db = self.databases[ row['db_id'] ]
        schema = str(db)
        question = f"{row['question']}  Hint: {row['evidence']}"
        return schema, question

    def generate_prompt(self, schema: str, question: str, **kwargs) -> str:
        """ Takes a question and a schema to generate the agent's SQL generation prompt """
        raise NotImplementedError
    
    def generate_text(self, prompts: list[str], cfg: SamplingParams, use_tqdm: bool = False) -> TextToSQLGenerationOutput:        
        outputs = self.llm.generate(prompts, sampling_params=cfg, use_tqdm=use_tqdm)

        input_prompts: list[str] = [output.prompt for output in outputs]
        raw_responses: list[str] = [output.outputs[0].text for output in outputs]
        parsed_sql:    list[str] = [self.auto_parse_sql(response) for response in raw_responses]
        n_in_tokens:   list[int] = [len(output.prompt_token_ids) for output in outputs]
        n_out_tokens:  list[int] = [len(output.outputs[0].token_ids) for output in outputs]

        return TextToSQLGenerationOutput(input_prompts, raw_responses, parsed_sql, n_in_tokens, n_out_tokens)
    
    def batched_generate(
        self, df: pd.DataFrame, cfg: SamplingParams, batch_size: int, 
        savename: str, evaluator_fn: Callable, **kwargs
    ) -> TextToSQLGenerationOutput:
        """ Generates SQL from a DataFrame of BIRD questions. Evaluates performance using evaluator_fn.
            Saves responses with savename as suffix.
            Kwargs passed on to process_bird_df().
        """
        input_prompts: list[str] = []
        raw_responses: list[str] = []
        parsed_sql:    list[str] = []
        n_in_tokens:   list[int] = []
        n_out_tokens:  list[int] = []
        
        for i, batch in enumerate(tqdm(batched(df, batch_size), desc=f'{savename} Generating SQL')):
            # Generate
            prompts: list[str] = [
                self.generate_prompt(*self.process_bird_df(idx, row, **kwargs))
                for idx, row in batch.iterrows()
            ]
            outputs = self.generate_text(prompts, cfg, use_tqdm=False)

            # Record responses
            input_prompts.extend(outputs.input_prompts)
            raw_responses.extend(outputs.raw_responses)
            parsed_sql.extend(outputs.parsed_sql)
            n_in_tokens.extend(outputs.n_in_tokens)
            n_out_tokens.extend(outputs.n_out_tokens)
            if savename:
                self.dump_to_json(f"{savename}_raw", raw_responses)
                self.dump_to_json(f"{savename}_clean", parsed_sql)
        
        final_output = TextToSQLGenerationOutput(input_prompts, raw_responses, parsed_sql, n_in_tokens, n_out_tokens)
        final_df = pd.concat(
            [df, final_output.as_dataframe(col_suffix=savename)], 
            axis=1,
        )
        if evaluator_fn:
            labels, report = evaluator_fn(final_df, self.databases, f'parsed_sql_{savename}')
            final_df[f'label_{savename}'] = labels
            with open(self.output_path/f'results_{savename}.txt', 'w') as f:
                f.write(report)
        final_df.to_json(self.output_path / f"df_batgen_{savename}.json", orient='records')
        return final_output

    def parse_with_regex(self, response: str) -> str:
        """ Extracts SQL from responses containing '''sql ... ''' using regex. """
        try:
            sql = re.search(r'```sql(.*?)```', response, re.DOTALL).group(1).strip()
        except AttributeError as e:
            sql = ''
        return sql
    
    def auto_parse_sql(self, response: str) -> str:
        """ Extracts SQL from responses containing '''sql ... ''' using regex. 
            If regex search fails, attempts to parse using LLM.
            Returns cleaned SQL or an empty string.
        """
        matched = self.parse_with_regex(response)
        if not matched:
            prompt = (
                "Please extract the SQL query from the text. Enclose your response within "
                "a ```sql <<your response here>> ``` code block. Exclude any additional "
                "text from your response, leaving only the SQL.\n\n"
                f"### Text:\n{response}\n\n"
                f"### SQL:\n"
            )
            raw_output = self.llm.generate(prompt, SamplingParams(temperature=0), use_tqdm=False)
            llm_parsed = raw_output[0].outputs[0].text
            matched = self.parse_with_regex(llm_parsed)
            if matched:
                print("Successfully parsed with LLM.")
            else:
                print("Failed to parse with LLM. Returning empty string.")
        return matched
    
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