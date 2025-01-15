import re
import sqlite3
from agents.TextToSQL import TextToSQL
from core.Model import LLM, GenerationConfig
from core.SQLiteDatabase import SQLiteDatabase

class SchemaAugmenter(TextToSQL):
    def augment_database(self, db: SQLiteDatabase, cfg: GenerationConfig, save: bool):
        """ Augments a database schema by adding descriptions as comments. Outputs augmented schema that is rich in detail while remaining 
            executable to the same effect. Enriched schema beneficial for TextToSQL as they provide additional context. 
         """
        db_id = db.db_id
        raw_schemas = db.raw_schema
        descriptions = db.descriptions
        full_schema = str(db)
        raw_responses = {}
        aug_schema = {}
        for (tbl, schema), (_, description) in zip(raw_schemas.items(), descriptions.items()):
            system_prompt = (f"Please answer any user questions regarding the following database {full_schema}.")
            user_prompt = (
                "I will give you an SQLite table schema and CSV file with database descriptions.\n"
                "1. Add the descriptions as comments to the schema and elaborate if required.\n"
                "2. Add a block comment to describe the purpose of the table.\n"
                "3. Add the relationship between this table and the others in the database to the description.\n"
                "Output the modified schema of this table. Ensure that it is valid SQLite.\n"
                f"### Schema:\n{schema}\n###Description CSV:\n{description}"
            )
            reply = self.llm(
                messages=[
                    {'role': 'system', "content": system_prompt},
                    {'role': 'user', "content": user_prompt}
                ],
                cfg=cfg
            )
            raw_responses[tbl] = reply
            aug_schema[tbl] = self.auto_parse_sql(reply)

            if not SchemaAugmenter.are_create_statements_equivalent(schema, aug_schema[tbl]):
                print(f"Augmented schema {tbl} of {db_id} contains discrepancies. Attempting repair...")
                fixer_prompt = (
                    "Please make the given schema the same as the sample. You should ensure the syntax is correct and the "
                    "ordering is exactly the same. Keep the comments untouched and output the modified SQLite schema.\n"
                    f"### Sample\n{schema}\n### Given Schema\n{aug_schema[tbl]}"
                )
                reply = self.llm(messages=[{'role': 'user', 'content': fixer_prompt}], cfg=cfg)
                fixed = self.auto_parse_sql(reply)

                if not SchemaAugmenter.are_create_statements_equivalent(fixed, aug_schema[tbl]):
                    print("Failed. Keeping it untouched.")
                else:
                    print("Success! Replacing with repaired version.")
                    aug_schema[tbl] = fixed              
        
        if save:
            self.dump_to_json(f'augmented/raw/{db_id}_aug_raw', raw_responses)
            self.dump_to_json(f'augmented/aug/{db_id}_aug', aug_schema)
        
        return raw_responses, aug_schema
    
    def augment_all(self, save: bool = True) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        """ Takes a dict of SQLiteDatabases and augments all of them. Caches schemas for use later. """
        raw_responses = {}
        aug_db_schemas = {}
        cfg = GenerationConfig(temperature=0.4, max_tokens=2048*3, num_ctx=4096*4)
        for db_id, db in self.databases.items():
            raw_responses[db_id], aug_db_schemas[db_id]  = self.augment_database(db, cfg, save)

        if save:
            self.dump_to_json(f'augmented/FULL_DB_aug_raw', raw_responses)
            self.dump_to_json(f'augmented/FULL_DB_ug', aug_db_schemas)

        return raw_responses, aug_db_schemas
    

    def are_create_statements_equivalent(create_stmt1, create_stmt2) -> bool:
        """ Checks if two table schemas are the same. Used to ensure augmented schema is the same as raw. """
        # Create an in-memory SQLite database
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        def replace_table_name(create_stmt, new_table_name):
            # Use regex to replace the table name in the CREATE TABLE statement
            return re.sub(r"CREATE TABLE\s+[`'\"]?[\w]+[`'\"]?", f"CREATE TABLE {new_table_name}", create_stmt, count=1)
        
        def preprocess_sql(sql):
            """Remove comments and extra whitespace from SQL."""
            # Remove multi-line comments
            sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
            # Remove single-line comments
            sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
            # Strip extra whitespace
            sql = re.sub(r"\s+", " ", sql).replace(" , ", ", ").strip()
            # Change table name so they are the same
            sql = replace_table_name(sql, 'tbl_test')
            return sql
        
        try:
            # Replace table names with unique names
            stmt1 = replace_table_name(preprocess_sql(create_stmt1), "test_table1")
            stmt2 = replace_table_name(preprocess_sql(create_stmt2), "test_table2")

            # Execute the modified CREATE statements
            cursor.execute(stmt1)
            cursor.execute(stmt2)

            # Fetch the schema definitions from sqlite_master
            cursor.execute("SELECT sql FROM sqlite_master WHERE name='test_table1'")
            schema1, = cursor.fetchall()[0]
            schema1 = preprocess_sql(schema1)
            cursor.execute("SELECT sql FROM sqlite_master WHERE name='test_table2'")
            schema2, = cursor.fetchall()[0]
            schema2 = preprocess_sql(schema2)

            # Compare normalized schema definitions
            return schema1.lower() == schema2.lower()
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return False
        finally:
            conn.close()

    