import os
from enum import Enum


class DbInfo:
    def __init__(self, name, is_file_db: bool = False, is_bigquery_db: bool = False):
        self.name = name
        self.is_file_db = is_file_db
        self.is_bigquery_db=is_bigquery_db


class DBType(Enum):
    MSSQL = DbInfo("mssql")
    Postgresql = DbInfo("postgresql")
    Spark = DbInfo("spark", True)
    Hive =DbInfo("hive")
    BigQuery =DbInfo("bigquery",False,True)
    
    def value(self):
        return self._value_.name

    def is_file_db(self):
        return self._value_.is_file_db
    def is_bigquery_db(self):
        return self._value_.is_bigquery_db

    @staticmethod
    def of_db_type(db_type: str):
        for item in DBType:
            if item.value() == db_type:
                return item
        return None

    @staticmethod
    def parse_file_db_name_from_path(db_type: str, local_db_path: str):
        """Parse out the database name of the embedded database from the file path"""
        base_name = os.path.basename(local_db_path)
        db_name = os.path.splitext(base_name)[0]
        if "." in db_name:
            db_name = os.path.splitext(db_name)[0]
        return db_type + "_" + db_name
