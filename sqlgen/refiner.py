import re
from vllm import LLM, SamplingParams


def parse_with_regex(response: str) -> str:
    """ Extracts SQL from responses containing '''sql ... ''' using regex. """
    try:
        sql = re.search(r'```sql(.*?)```', response, re.DOTALL).group(1).strip()
    except AttributeError as e:
        sql = ''
    return sql


def llm_parse_sql(llm: LLM, response: str) -> str:
    """ Extracts SQL from responses containing '''sql ... ''' using regex. 
        If regex search fails, attempts to parse using LLM.
        Returns cleaned SQL or an empty string.
    """
    matched = parse_with_regex(response)
    if not matched:
        prompt = (
            "Please extract the SQL query from the text. Enclose your response within "
            "a ```sql <<your response here>> ``` code block. Exclude any additional "
            "text from your response, leaving only the SQL.\n\n"
            f"### Text:\n{response}\n\n"
            f"### SQL:\n"
        )
        raw_output = llm.generate(prompt, SamplingParams(temperature=0), use_tqdm=False)
        llm_parsed = raw_output[0].outputs[0].text
        matched = parse_with_regex(llm_parsed)
        if matched:
            print("Successfully parsed with LLM.")
        else:
            print("Failed to parse with LLM. Returning empty string.")
    return matched


def check_validity_of_columns(columns: list[str]) -> bool:
    raise NotImplementedError