import logging
import sqlite3
from sqlite3 import Connection, Error
from typing import Optional, List, Tuple, Any


class HistorySQLite:
    def __init__(self, db_file: str) -> None:
        self.db_file: str = db_file
        self.conn: Optional[Connection] = None
        self.connect()

    def connect(self) -> None:
        try:
            self.conn = sqlite3.connect(self.db_file)
        except Error as e:
            logging.error(f"Error connecting to database: {e}")

    def close_connection(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_table(self) -> None:
        if not self.conn:
            logging.error("No connection available.")
            return
        try:
            sql_create_history_table = """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                variant TEXT NOT NULL,
                model_version TEXT NOT NULL,
                input_data TEXT NOT NULL,
                start_time INTEGER NOT NULL,
                end_time INTEGER NOT NULL
            );
            """
            cursor = self.conn.cursor()
            cursor.execute(sql_create_history_table)
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error creating table: {e}")

    def insert_history(
        self,
        run_id: str,
        variant: str,
        model_version: str,
        input_data: str,
        start_time: int,
        end_time: int
    ) -> None:
        if not self.conn:
            logging.error("No connection available.")
            return
        try:
            sql_insert_history = """
            INSERT INTO history (run_id, variant, model_version, input_data, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?);
            """
            cursor = self.conn.cursor()
            cursor.execute(sql_insert_history, (run_id, variant, model_version, input_data, start_time, end_time))
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error inserting history: {e}")
            self.conn.rollback()

    def fetch_history(self, model_version: str|None = None) -> List[Tuple[Any, ...]]:
        if not self.conn:
            logging.error("No connection available.")
            return []
        try:
            if model_version is not None:
                sql_fetch_history = "SELECT * FROM history WHERE model_version = ?;"
                cursor = self.conn.cursor()
                cursor.execute(sql_fetch_history, (model_version,))
            else:
                sql_fetch_history = "SELECT * FROM history;"
    
            cursor = self.conn.cursor()
            cursor.execute(sql_fetch_history)
            rows = cursor.fetchall()
            return rows
        except sqlite3.Error as e:
            logging.error(f"Error fetching history: {e}")
            return []