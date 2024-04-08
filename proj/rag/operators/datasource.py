from typing import Any

from proj.core.interface.operators.retriever import RetrieverOperator
from proj.datasource.rdbms.base import RDBMSDatabase
from proj.rag.summary.rdbms_db_summary import _parse_db_summary


class DatasourceRetrieverOperator(RetrieverOperator[Any, Any]):
    def __init__(self, connection: RDBMSDatabase, **kwargs):
        super().__init__(**kwargs)
        self._connection = connection

    def retrieve(self, input_value: Any) -> Any:
        summary = _parse_db_summary(self._connection)
        return summary
