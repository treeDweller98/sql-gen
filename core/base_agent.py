import re
import json
from pathlib import Path
from abc import ABC
from typing import Optional, Callable, Sequence
import datetime
from tqdm import tqdm
import pandas as pd
from vllm import LLM, SamplingParams
from core.dbhandler import SQLiteDatabase


def batched(sequence: Sequence, n: int=1):
    # Same as `from itertools import batched`
    l = len(sequence)
    for ndx in range(0, l, n):
        yield sequence[ndx:min(ndx + n, l)]


def dump_to_json(output_dir: Path, filename: str, obj: object) -> None:
    """ Dumps a list of objects to output_dir/filename.json; use for keeping backups. """
    filepath = output_dir / f"{filename}.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


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
        """ For use with single requests """
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
    ) -> None:
        """ Attributes
            ----------
                llm: LLM
                    Text generation model for offline generation
                databases: dict[str, SQLiteDatabase]
                    Dictionary of SQLiteDatabases indexed by db_id
        """
        self.llm = llm
        self.databases = databases

    def process_question(self, idx: int, row: pd.DataFrame, **kwargs) -> tuple:
        """ Takes a row of a DataFrame of dataset questions. 
            Processes and returns necessary columns required by this Agent's generate_response(). 
            Output tuple must be unpackable as parameters to generate_response().
        """
        db = self.databases[ row['db_id'] ]
        schema = str(db)
        question = row['question']
        return schema, question

    def generate_prompt(self, schema: str, question: str, **kwargs) -> str:
        """ Takes a question and a schema to generate the agent's SQL generation prompt """
        raise NotImplementedError
    
    def generate_text(self, prompts: list[str] | list[list[dict]], cfg: SamplingParams, use_tqdm: bool = False) -> TextToSQLGenerationOutput:
        if all(isinstance(item, str) for item in prompts):
            messages = [[{'role':'user', 'content': prompt}] for prompt in prompts]
        elif all(isinstance(item, list) and all(isinstance(subitem, dict) for subitem in item) for item in prompts):
            messages = prompts

        outputs = self.llm.chat(messages, sampling_params=cfg, use_tqdm=use_tqdm)
        input_prompts: list[str] = [output.prompt for output in outputs]
        raw_responses: list[str] = [output.outputs[0].text for output in outputs]
        parsed_sql:    list[str] = [self.parse_with_regex(response) for response in raw_responses]
        n_in_tokens:   list[int] = [len(output.prompt_token_ids) for output in outputs]
        n_out_tokens:  list[int] = [len(output.outputs[0].token_ids) for output in outputs]

        return TextToSQLGenerationOutput(input_prompts, raw_responses, parsed_sql, n_in_tokens, n_out_tokens)
    
    def parse_with_regex(self, response: str) -> str:
        """ Extracts SQL from responses containing '''sql ... ''' using regex. """
        try:
            if "```sqlite" in response:
                response = response.replace("```sqlite", "```sql")
            sql = re.search(r'```sql(.*?)```', response, re.DOTALL).group(1).strip()
        except AttributeError as e:
            sql = ''
        return sql
    
    def batched_generate(
        self, df: pd.DataFrame, cfg: SamplingParams, batch_size: int,
        output_dir: Path, savename: str, evaluator_fn: Optional[Callable] = None, 
        **kwargs
    ) -> tuple[TextToSQLGenerationOutput, Optional[list[bool]]]:
        """ Generates SQL from a DataFrame of BIRD questions. 
            Evaluates performance using evaluator_fn.
            Saves responses with savename as suffix in output_dir.
            Kwargs passed on to process_question().
            Returns TextGenerationOutput, along with labels if evaluate_fn given
        """
        start_time = datetime.datetime.now()
        input_prompts: list[str] = []
        raw_responses: list[str] = []
        parsed_sql:    list[str] = []
        n_in_tokens:   list[int] = []
        n_out_tokens:  list[int] = []
        
        for i, batch in enumerate(tqdm(batched(df, batch_size), desc=f'{savename} Generating SQL')):
            # Generate
            prompts: list[str] = [
                self.generate_prompt(*self.process_question(idx, row, **kwargs))
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
                dump_to_json(output_dir, f"{savename}_raw", raw_responses)
                dump_to_json(output_dir, f"{savename}_clean", parsed_sql)
        
        final_output = TextToSQLGenerationOutput(input_prompts, raw_responses, parsed_sql, n_in_tokens, n_out_tokens)
        final_df = pd.concat(
            [df, final_output.as_dataframe(col_suffix=savename)], 
            axis=1,
        )
        print(f"batched_generate() completed in {datetime.datetime.now() - start_time}")
        if evaluator_fn:
            labels, report = evaluator_fn(final_df, self.databases, f'parsed_sql_{savename}')
            final_df[f'label_{savename}'] = labels
            with open(output_dir/f'results_{savename}.txt', 'w') as f:
                f.write(report)
        else:
            labels = None
        final_df.to_json(output_dir / f"df_batgen_{savename}.json", orient='records')
        return final_output, labels
