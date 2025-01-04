import sqlite3
from pathlib import Path
from typing import Optional

class SQLiteDatabase:
    """ Class for dealing with sqlite3 databases. Provides SQL execution capabilities and access to schema"""
    def __init__(self, db_id: str, input_path: Path, use_cached_schema: Optional[Path] = None) -> None:
        """ Attributes
            ----------
                db_id: str
                    name of database; database must exist in input_path/db_id/db_id.sqlite
                input_path: Path
                    parent directory of database folder e.g. input_path/db_id/db_id.sqlite 
                cursor: sqlite3.Cursor
                    connected cursor to the database, used to execute all queries
                schema: dict[str, str]
                    ...

                raw_schema: dict[str, str]
                    unaugmented, plain db schemas indexed by table_name, read from db_id.sqlite
                descriptions: dict[str, str]
                    Table descriptions, indexed by table_name, read from table_name.csv 
                    which exist in input_path/db_id/database_description/

                use_cached_schemas: Path | None
                    use pre-generated schema stored in input_path/db_id/db_id_augschema.json
                    instead of raw_schema
        """
        self.db_id = db_id
        self.input_path = input_path

        self.cursor: sqlite3.Cursor = self.__get_db_cursor()
        self.raw_schema: dict[str, str] = self.__fetch_raw_schema()
        self.descriptions: dict[str, str] = self.__fetch_db_descriptions()

        if use_cached_schema:
            self.schema = ... # read from json
        else:
            self.schema = self.raw_schema

    def __getitem__(self, table_name: str):
        return self.schemas[table_name]
    
    def __str__(self):
        return "\n".join( list(self.schema.values()) )
    
    def run_query(self, sql: str) -> list:
        """ Executes SQL query and returns rows. """
        rows = self.cursor.execute(sql).fetchall()
        return rows

    def __get_db_cursor(self) -> sqlite3.Cursor:
        """ Connects to db and returns a read-only cursor. """
        db_path = (self.input_path / self.db_id / self.db_id).with_suffix('.sqlite')
        connection = sqlite3.connect(db_path, uri=True)
        cursor = connection.cursor()
        return cursor
    
    def __fetch_raw_schema(self) -> dict[str, str]:
        """ Returns a dict of schema of all tables in a .sqlite database indexed by table name """
        tables = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        schemas: dict[str, str] = {}
        for table, in tables:
            if table != "sqlite_sequence":
                schema = self.cursor.execute("SELECT sql FROM sqlite_master WHERE name=?;", [table]).fetchone()[0]
                schemas[table] = schema
        return schemas
    
    def __fetch_db_descriptions(self) -> dict[str, str]:
        descriptions = {}
        for table in self.raw_schema.keys():
            filename = (self.input_path / self.db_id / 'database_description' / table).with_suffix('.csv')
            with open(filename, 'r', errors='ignore') as file:
                descriptions[table] = file.read()
        return descriptions
    


if __name__ == '__main__':
    db_id = 'formula_1'
    # db_id = 'thrombosis_prediction'
    input_path = Path('data/bird-minidev/dev_databases')
    db = SQLiteDatabase(db_id, input_path)
    
    for name, description in db.descriptions.items():
        print('-' * 25)
        print(name.capitalize())
        print('-' * 25)
        print(description)
        print('.' * 25)
        print('.' * 25)
        print('\n')
    print(db)



# ### Database Utility Functions ###
# def get_db_cursor(db_id: str) -> sqlite3.Cursor:
#     """ Connects to db and returns a cursor. """
#     db_path = (INPUT_PATH / 'dev_databases/' / db_id / db_id).with_suffix('.sqlite')
#     connection = sqlite3.connect(db_path)
#     cursor = connection.cursor()
#     return cursor


# def fetch_BIRD_schemas(db_names: list[str]) -> dict[str, str]:
#     """ Returns a dictionary of BIRD db_schemas indexed by db_names. """
    
#     def fetch_schema(db_id: str) -> str:
#         """Returns the schema of all tables in a .sqlite database. """
#         cursor = get_db_cursor(db_id)
#         cursor = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#         tables = cursor.fetchall()
        
#         schemas: list[str] = []
#         for table in tables:
#             table_name, = table
#             if table_name != "sqlite_sequence":
#                 cursor = cursor.execute("SELECT sql FROM sqlite_master WHERE name=?;", [table_name])
#                 schema = cursor.fetchone()[0]
#                 schemas.append(schema)

#         return "\n".join(schemas)

#     return {db_id: fetch_schema(db_id) for db_id in db_names}