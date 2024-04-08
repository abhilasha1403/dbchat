from typing import Any, Iterable, Optional
from urllib.parse import quote
from urllib.parse import quote_plus as urlquote

from proj.datasource.rdbms.base import RDBMSDatabase
class BQConnect(RDBMSDatabase):
    driver = "bigquery"
    db_type = "bigquery"
    db_dialect = "bigquery"

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
        
         db_url: str = (
            f'{cls.driver}://{host}/{db_name}?credentials_path={pwd}'
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