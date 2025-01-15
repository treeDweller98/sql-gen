import json
import sqlite3
from pathlib import Path
from typing import Optional
from func_timeout import func_timeout, FunctionTimedOut

class SQLiteDatabase:
    """ Class for dealing with sqlite3 databases. Provides SQL execution capabilities and access to schema"""
    def __init__(self, db_id: str, input_path: Path, exec_timeout: float = 30.0, use_cached_schema: Optional[Path] = None) -> None:
        """ Attributes
            ----------
                db_id: str
                    name of database; database must exist in input_path/db_id/db_id.sqlite
                input_path: Path
                    parent directory of database folder
                exec_timeout: float
                    maximum number of seconds for query to return a result; aborts returning [(err),]
                schema: dict[str, str]
                    either raw_schema or read from json in path 

                raw_schema: dict[str, str]
                    unaugmented, plain db schemas indexed by table_name, read from db_id.sqlite
                descriptions: dict[str, str]
                    Table descriptions, indexed by table_name, read from table_name.csv 
                    which exist in input_path/db_id/database_description/

                use_cached_schemas: Path | None
                    use pre-generated schema stored in path/to/aug.json provided
                    instead of raw_schema. File contains dict of db_id: schema.
        """
        self.db_id = db_id
        self.input_path = input_path
        self.exec_timeout = exec_timeout

        self.raw_schema: dict[str, str] = self.__fetch_raw_schema()
        self.descriptions: dict[str, str] = self.__fetch_db_descriptions()

        if use_cached_schema:
            with open(use_cached_schema, 'r') as f:
                self.schema = json.load(f)[db_id]
                print(type(self.schema))
        else:
            self.schema = self.raw_schema

    def __getitem__(self, table_name: str):
        """ Return the schema of a table in the database. """
        return self.schema[table_name] 
    
    def __str__(self):
        """ Returns the database schema as a human-readable/executable string. """
        return "\n\n".join( list(self.schema.values()) )
    
    def run_query(self, sql: str) -> list[tuple]:
        """ Executes SQL query and fetches all rows. """
        try:
            def execute_sql():
                db_path = (self.input_path / self.db_id / self.db_id).with_suffix('.sqlite')
                with sqlite3.connect(db_path, uri=True) as conn:
                    rows = conn.execute(sql).fetchall()
                return rows

            rows = func_timeout(timeout=self.exec_timeout, func=execute_sql)
        except FunctionTimedOut as timeout_error:
            rows = [("Error: timedout", )]
        return rows
    
    def __fetch_raw_schema(self) -> dict[str, str]:
        """ Returns a dict of schema of all tables in a .sqlite database indexed by table name """
        tables = self.run_query("SELECT name FROM sqlite_master WHERE type='table';")
        schemas: dict[str, str] = {}
        for table, in tables:
            if table != "sqlite_sequence":
                schema, = self.run_query(f"SELECT sql FROM sqlite_master WHERE name='{table}';")[0]
                schemas[table] = schema
        return schemas
    
    def __fetch_db_descriptions(self) -> dict[str, str]:
        """ Returns a dict of database_descriptions from each table_name.csv as strings  """
        def case_insensitive_file_reader(filepath: Path):
            content = f'Descriptions file for table at {filepath} does not exist.'
            if filepath.exists():
                with open(filepath, 'r', errors='ignore') as file:
                    content = file.read()
            else:
                file_stem = filepath.stem
                candidate_stems = [
                    stem for stem in (
                        file_stem.capitalize(), file_stem.title(), file_stem.upper(), file_stem.lower()
                    )
                    if filepath.with_stem(stem).exists()
                ]
                if candidate_stems:
                    file_stem = candidate_stems[0]
                    filepath = filepath.with_stem(file_stem)
                    content = case_insensitive_file_reader(filepath)
            return content
        
        descriptions = {}
        for table in self.raw_schema.keys():
            filepath = (self.input_path / self.db_id / 'database_description' / table).with_suffix('.csv')
            descriptions[table] = case_insensitive_file_reader(filepath)

        return descriptions
    


if __name__ == '__main__':
    db_id = 'formula_1'
    # db_id = 'thrombosis_prediction'
    input_path = Path('data/bird-minidev/dev_databases')
    db = SQLiteDatabase(db_id, input_path)
    rows = db.run_query(
        "SELECT T2.driverRef "
        "FROM qualifying AS T1 INNER JOIN drivers AS T2 "
        "ON T2.driverId = T1.driverId "
        "WHERE T1.raceId = 20 "
        "ORDER BY T1.q1 DESC "
        "LIMIT 5"
    )
    print(rows)

    # for name, description in db.descriptions.items():
    #     print('-' * 25)
    #     print(name.capitalize())
    #     print('-' * 25)
    #     print(description)
    #     print('.' * 25)
    #     print('.' * 25)
    #     print('\n')
    # print(db)

    ## Run in jupyter
    # db_names: list[str] = [f.name for f in input_path.iterdir()]
    # databases: dict[str, SQLiteDatabase] = {
    #     db_id: SQLiteDatabase(db_id, input_path) 
    #     for db_id in db_names
    # }
    # for db_id, db in databases.items():
    #     print(db_id)
    #     display(db.raw_schema)
    #     display(db.descriptions)