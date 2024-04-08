from typing import Any, Iterable, Optional
from urllib.parse import quote
from urllib.parse import quote_plus as urlquote

from sqlalchemy import text, inspect, create_engine

from proj.datasource.rdbms.base import RDBMSDatabase


class HiveConnect(RDBMSDatabase):
    driver = "hive"
    db_type = "hive"
    db_dialect = "hive"

    @classmethod
    def from_uri_db(
        cls,
        host: str,
        port: int,
        user: str,
        pwd: str,
        db_name: str,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> RDBMSDatabase:
        # db_url: str = (
        #     f'{cls.driver}://{host}:{port}/{db_name}?username={user}&password={pwd}'
        # )
        db_url: str = (
            f"hive://localhost:10000/dvdrental"
        )
        return cls.from_uri(db_url, engine_args, **kwargs)

  
    def get_users(self):
        return []

    def get_grants(self):
        return []
    def get_charset(self):
        return "UTF-8"
    def get_collation(self):
        """Get collation."""
        return "UTF-8"