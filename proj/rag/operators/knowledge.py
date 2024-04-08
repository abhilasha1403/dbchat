from typing import Any, List, Optional

from proj.core.awel import MapOperator
from proj.core.awel.task.base import IN
from proj.rag.knowledge.base import Knowledge, KnowledgeType
from proj.rag.knowledge.factory import KnowledgeFactory


class KnowledgeOperator(MapOperator[Any, Any]):
    """Knowledge Operator."""

    def __init__(
        self, knowledge_type: Optional[KnowledgeType] = KnowledgeType.DOCUMENT, **kwargs
    ):
        """Init the query rewrite operator.
        Args:
            knowledge_type: (Optional[KnowledgeType]) The knowledge type.
        """
        super().__init__(**kwargs)
        self._knowledge_type = knowledge_type

    async def map(self, datasource: IN) -> Knowledge:
        """knowledge operator."""
        return await self.blocking_func_to_async(
            KnowledgeFactory.create, datasource, self._knowledge_type
        )
