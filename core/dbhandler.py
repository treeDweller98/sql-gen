### core/dbhandler.py
import json
import sqlite3
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout

# TODO: Add M-Schema


class SQLiteDatabase:
    ERROR_TIMEOUT_RESULT: list[tuple[str]] = [("Error: timedout",)]

    """ Handler class for sqlite3 databases. Provides SQL execution capabilities and access to schema"""

    def __init__(
        self, db_id: str, input_dir: Path, exec_timeout: float = 30.0, use_cached_schema: Path | None = None
    ) -> None:
        """Attributes
        ----------
            db_id: str
                name of database; database must exist in input_dir/db_id/db_id.sqlite
            input_dir: Path
                parent directory of database folder
            db_path: Path
                full path of the db_id.sqlite file
            exec_timeout: float
                maximum number of seconds for query to return a result; aborts returning [(err),]
            schema: dict[str, str]
                either raw_schema or read from json in path

            raw_schema: dict[str, str]
                unaugmented, plain db schemas indexed by table_name, read from db_id.sqlite
            descriptions: dict[str, str]
                Table descriptions, indexed by table_name, read from table_name.csv
                which exist in input_dir/db_id/database_description/
            table_columns: dict[str, set[str]]
                Set of column names of each table, indexed by table name

            use_cached_schemas: Path | None
                use pre-generated schema stored in path/to/aug.json provided
                instead of raw_schema. File must map db_id: schema.
        """
        self.db_id = db_id
        self.input_dir = input_dir
        self.db_path = (self.input_dir / self.db_id / self.db_id).with_suffix(".sqlite")
        self.exec_timeout = exec_timeout

        self.raw_schema: dict[str, str] = self.__fetch_raw_schema()
        self.descriptions: dict[str, str] = self.__fetch_db_descriptions()
        self.table_columns: dict[str, set[str]] = self.__fetch_table_columns()

        if use_cached_schema:
            with open(use_cached_schema, "r") as f:
                self.schema = json.load(f)[db_id]
        else:
            self.schema = self.raw_schema

    def __getitem__(self, table_name: str):
        """Return the schema of a table in the database."""
        return self.schema[table_name]

    def __str__(self):
        """Returns the database schema as a human-readable/executable string."""
        return "\n".join(list(self.schema.values()))

    def run_query(self, sql: str, timeout: float | None = None) -> list[tuple]:
        """Executes SQL query and fetches all rows."""
        try:

            def execute_sql():
                with sqlite3.connect(self.db_path, uri=True) as conn:
                    rows = conn.execute(sql).fetchall()
                return rows

            rows = func_timeout(timeout=(timeout or self.exec_timeout), func=execute_sql)
        except FunctionTimedOut as timeout_error:
            rows = self.ERROR_TIMEOUT_RESULT
        return rows

    def __fetch_raw_schema(self) -> dict[str, str]:
        """Returns a dict of schema of all tables in a .sqlite database indexed by table name"""
        tables = self.run_query("SELECT name FROM sqlite_master WHERE type='table';")
        schemas: dict[str, str] = {}
        for (table,) in tables:
            if table != "sqlite_sequence":
                (schema,) = self.run_query(f"SELECT sql FROM sqlite_master WHERE name='{table}';")[0]
                schemas[table] = schema
        return schemas

    def __fetch_table_columns(self) -> dict[str, set[str]]:
        """Returns a list of column names of each table indexed by table names"""
        table_names = list(self.raw_schema.keys())
        table_column_names = {}
        for table in table_names:
            columns = [col[1] for col in self.run_query(f'PRAGMA table_info("{table}");')]
            table_column_names[table] = set(columns)
        return table_column_names

    def __fetch_db_descriptions(self) -> dict[str, str]:
        """Returns a dict of database_descriptions from each table_name.csv as strings"""

        def case_insensitive_file_reader(filepath: Path):
            content = f"Descriptions file for table at {filepath} does not exist."
            if filepath.exists():
                with open(filepath, "r", errors="ignore") as file:
                    content = file.read()
            else:
                file_stem = filepath.stem
                candidate_stems = [
                    stem
                    for stem in (file_stem.capitalize(), file_stem.title(), file_stem.upper(), file_stem.lower())
                    if filepath.with_stem(stem).exists()
                ]
                if candidate_stems:
                    file_stem = candidate_stems[0]
                    filepath = filepath.with_stem(file_stem)
                    content = case_insensitive_file_reader(filepath)
            return content

        descriptions = {}
        for table in self.raw_schema.keys():
            filepath = (self.input_dir / self.db_id / "database_description" / table).with_suffix(".csv")
            descriptions[table] = case_insensitive_file_reader(filepath)

        return descriptions


if __name__ == "__main__":
    input_dir = Path(f"data/bird-minidev")
    bird_question_filename = "dev.json"
    db_foldername = "dev_databases"
    db_exec_timeout = 30.0
    use_cached_schema = False

    db_names: list[str] = [f.name for f in (input_dir / db_foldername).iterdir()]

    databases: dict[str, SQLiteDatabase] = {
        db_id: SQLiteDatabase(db_id, (input_dir / db_foldername), db_exec_timeout, use_cached_schema)
        for db_id in db_names
    }
