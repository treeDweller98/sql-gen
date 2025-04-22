import re
import sqlite3
from core.dbhandler import SQLiteDatabase


### instantiate _db from train.py first
_db: dict[str, SQLiteDatabase] = None

def set_db(db: dict[str, SQLiteDatabase]) -> None:
    global _db
    _db = db

def _get_db() -> dict[str, SQLiteDatabase]:
    if _db is None:
        raise RuntimeError("Database not initialized in reward_utils")
    return _db


### Utility functions used by reward functiosn
def parse_with_regex(response: str) -> str:
    """ Extracts SQL from responses containing ```sql ... ``` using regex. """
    try:
        sql = re.search(r'```sql(.*?)```', response, re.DOTALL).group(1).strip()
    except AttributeError as e:
        sql = ''
    return sql


def is_sql_same(db_id: str, pred_sql: str, gold_sql: str) -> bool:
    """ Executes SQL queries and returns True if outputs match, with no operation errors """
    try:
        db = _get_db()[db_id]
        res_gold = db.run_query(gold_sql)     # assumes valid SQLite
        res_pred = db.run_query(pred_sql)
    except sqlite3.OperationalError as e:
        return False
    else:
        return set(res_gold) == set(res_pred)

def is_sql_valid(db_id: str, sql: str) -> bool:
    """ Returns True if SQL executes without OperationalErrors """
    try:
        db = _get_db()[db_id]
        _ = db.run_query(sql)
        return True
    except sqlite3.OperationalError as e:
        return False
    

def extract_table_column_names(sql: str) -> set[str]:
    """ Uses regular expressions to chip away keywords etc from SQL queries to
        leave behind only table and column names. 
    """
    SQLITE_KEYWORDS = (
        # 147 keywords taken from https://sqlite.org/lang_keywords.html
        r"\b(ABORT|ACTION|ADD|AFTER|ALL|ALTER|ALWAYS|ANALYZE|AND|AS|ASC|"
        r"ATTACH|AUTOINCREMENT|BEFORE|BEGIN|BETWEEN|BY|CASCADE|CASE|CAST|"
        r"CHECK|COLLATE|COLUMN|COMMIT|CONFLICT|CONSTRAINT|CREATE|CROSS|"
        r"CURRENT|CURRENT_DATE|CURRENT_TIME|CURRENT_TIMESTAMP|DATABASE|"
        r"DEFAULT|DEFERRABLE|DEFERRED|DELETE|DESC|DETACH|DISTINCT|DO|DROP|"
        r"EACH|ELSE|END|ESCAPE|EXCEPT|EXCLUDE|EXCLUSIVE|EXISTS|EXPLAIN|"
        r"FAIL|FILTER|FIRST|FOLLOWING|FOR|FOREIGN|FROM|FULL|GENERATED|GLOB|"
        r"GROUP|GROUPS|HAVING|IF|IGNORE|IMMEDIATE|IN|INDEX|INDEXED|INITIALLY|"
        r"INNER|INSERT|INSTEAD|INTERSECT|INTO|IS|ISNULL|JOIN|KEY|LAST|LEFT|"
        r"LIKE|LIMIT|MATCH|MATERIALIZED|NATURAL|NO|NOT|NOTHING|NOTNULL|NULL|"
        r"NULLS|OF|OFFSET|ON|OR|ORDER|OTHERS|OUTER|OVER|PARTITION|PLAN|PRAGMA|"
        r"PRECEDING|PRIMARY|QUERY|RAISE|RANGE|RECURSIVE|REFERENCES|REGEXP|"
        r"REINDEX|RELEASE|RENAME|REPLACE|RESTRICT|RETURNING|RIGHT|ROLLBACK|"
        r"ROW|ROWS|SAVEPOINT|SELECT|SET|TABLE|TEMP|TEMPORARY|THEN|TIES|TO|"
        r"TRANSACTION|TRIGGER|UNBOUNDED|UNION|UNIQUE|UPDATE|USING|VACUUM|"
        r"VALUES|VIEW|VIRTUAL|WHEN|WHERE|WINDOW|WITH|WITHOUT|"
        # CAST data types
        r"NONE|TEXT|INTEGER|REAL|NUMERIC|"
        # TODO: add more common functions
        r"AVG|COUNT|CONCAT|MAX|MIN|SUM|TOTAL|POW|POWER|EXP|SQRT|MOD|FLOOR|CEIL|"
        r"LENGTH|LOWER|UPPER|TRIM|SUBSTR|FORMAT|PRINTF|TYPEOF|"
        r"DATE|TIME|DATETIME|JULIANDAY|UNIXEPOCH|STRFTIME|TIMEDIFF|"
        r"COALESCE|IIF|NULLIF|ABS|ROUND|"
        r"RANDOM|IFNULL|INSTR)\b"
    )
    text = sql
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)         # multi-line comments
    text = re.sub(r"--.*?$", " ", text, flags=re.MULTILINE)         # single-line comments
    text = re.sub(r"([`'\"]).*?\1", " ", text)                      # anything quoted using ` ' "
    text = re.sub(r"\b-?\d+(?:\.\d+)?\b", " ", text)                # digits, decimals
    text = re.sub(r"\W+", " ", text)                                # all non-alphanumerics
    text = re.sub(SQLITE_KEYWORDS, " ", text, flags=re.IGNORECASE)  # removes all sqlite keywords
    tokens = text.split()
    
    aliases  = re.findall(r"[a-zA-Z0-9_]+(?=\.)", sql)          # T1.some_col -> T1.
    aliases += re.findall(r"(?i)(?<=AS\s)[a-zA-Z0-9_]+", sql)   # table AS T_alias -> T_alias
    # TODO: Edge cases
    # needs a regex for aliases defined sequentially "SELECT c1 as a1, c2 a2, c3 AS a3..." -> a2
    # needs a regex for aliases defined without AS "SELECT * FROM Table T1 ..." -> T1
    # restore content in quotes if they match tables and column names 
    # quotation-mark insensitive string comparison for restored names
    tables_columns = set(tokens) - set(aliases) - set('_')
    return tables_columns


def case_insensitive_diff(set_1: set[str], set_2: set[str]) -> set[str]:
    return set(map(str.lower, set_1)) - set(map(str.lower, set_2))

def unroll_dict_to_set(dictionary: dict) -> set:
    return set(dictionary.keys()) | set(val for values in dictionary.values() for val in values)

def extract_hallucinated_table_col(db_id: str, candidates: set[str]) -> set[str]:
    """ Compares candidates to database original table and column names to find non-existent/hallucinated names """
    db = _get_db()[db_id]
    og_table_col = unroll_dict_to_set(db.table_columns)
    hallucinated = case_insensitive_diff(candidates, og_table_col)
    return hallucinated




if __name__ == '__main__':
    from pathlib import Path
    import pandas as pd
    from core.dbhandler import SQLiteDatabase

    input_path  = Path(f'data/bird-minidev')
    bird_question_filename = 'dev.json'
    db_foldername = 'dev_databases'
    db_exec_timeout = 30.0
    use_cached_schema = False  

    df = pd.read_json(input_path / bird_question_filename)
    databases: dict[str, SQLiteDatabase] = {
        db_id: SQLiteDatabase(db_id, (input_path / db_foldername), db_exec_timeout, use_cached_schema) 
        for db_id in [f.name for f in (input_path / db_foldername).iterdir()]
    }
    set_db(databases)

    db_id = 'debit_card_specializing'
    raw_response = "1. **Understand the Task**: The task is to find the ratio of customers who use EUR as their currency against those who use CZK. This involves counting the number of customers for each currency and then dividing these counts.\n\n2. **Identify Relevant Table and Columns**: From the schema, the `customers` table contains the `Currency` column which is relevant for this query.\n\n3. **Formulate the SQL Query**:\n   - Use the `COUNT` function to count the number of customers for each currency.\n   - Use a `CASE` statement within the `COUNT` function to differentiate between EUR and CZK.\n   - Divide the count of EUR customers by the count of CZK customers to get the desired ratio.\n\n4. **Construct the Query**:\n   - Start with the `SELECT` clause to specify the calculation.\n   - Use `CAST` to ensure that the division results in a real number.\n   - Use `COUNT(CASE WHEN Currency = 'EUR' THEN 1 ELSE NULL END)` to count EUR customers.\n   - Use `COUNT(CASE WHEN Currency = 'CZK' THEN 1 ELSE NULL END)` to count CZK customers.\n   - Divide these two counts to get the ratio.\n\n5. **Finalize the Query**:\n   - Ensure the query is correctly formatted and all necessary components are included.\n\nTherefore, the final SQL query is:\n\n```sql\nSELECT CAST(COUNT(CASE WHEN Currency = 'EUR' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(CASE WHEN Currency = 'CZK' THEN 1 ELSE NULL END) FROM customers\n```"
    parsed_sql   = "SELECT CAST(COUNT(CASE WHEN Currency = 'EUR' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(CASE WHEN Currency = 'CZK' THEN 1 ELSE NULL END) FROM customers"
    parsed_inval = "SELECTYYY CAST(COUNT(CASE WHEN Currency = 'EUR' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(CASE WHEN Currency = 'CZK' THEN 1 ELSE NULL END) FROM customers"
    gold = "SELECT SUM(Consumption) FROM yearmonth WHERE CustomerID = 6 AND Date BETWEEN '201308' AND '201311'"
    pred = "SELECT SUM(Consumption) AS TotalConsumption\nFROM yearmonth\nWHERE CustomerID = 6 AND Date BETWEEN '201308' AND '201311'"

    assert parse_with_regex(raw_response) == parsed_sql
    assert is_sql_valid(db_id, parsed_sql) == True
    assert is_sql_valid(db_id, parsed_inval) == False
    assert is_sql_same(db_id, pred, gold) == True
    assert is_sql_same(db_id, parsed_sql, gold) == False
    assert is_sql_same(db_id, parsed_inval, parsed_sql) == False

    assert case_insensitive_diff({"Apple", "Banana", "Cherry"}, {"apple", "banana"}) == {"cherry"}      # basic
    assert case_insensitive_diff({"ONE", "TWO"},  {"one", "two"}) == set()                              # all same
    assert case_insensitive_diff({"Cat", "Dog"}, {"Bird", "Fish"}) == {"cat", "dog"}                    # disjoint
    assert case_insensitive_diff(set(), set()) == set()                                                 # empty
    assert case_insensitive_diff({"Hello"}, set()) == {"hello"}
    assert case_insensitive_diff(set(), {"Hello"}) == set()

    assert unroll_dict_to_set({"a": [1, 2], "b": [3]}) == {"a", "b", 1, 2, 3}
    assert unroll_dict_to_set({}) == set()
    assert unroll_dict_to_set({"x": [1, 2], "y": [2, 3], "z": [1]}) == {"x", "y", "z", 1, 2, 3}
    assert unroll_dict_to_set({1: ['a'], 'b': [2.5, 1]}) == {1, 'b', 'a', 2.5}


    assert extract_table_column_names(gold) == {'yearmonth', 'CustomerID', 'Consumption'}
    assert extract_table_column_names(pred) == {'yearmonth', 'CustomerID', 'Consumption'}
    assert extract_table_column_names(parsed_inval) == {'SELECTYYY', 'customers', 'Currency'}

    for sql in [parsed_sql, parsed_inval, gold, pred]:
        assert extract_hallucinated_table_col(db_id, extract_table_column_names(sql)) == set() or {'selectyyy'}

    print("Success!")