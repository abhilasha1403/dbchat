from typing import Any


def _import_pgvector() -> Any:
    from proj.storage.vector_store.pgvector_store import PGVectorStore

    return PGVectorStore


def _import_chroma() -> Any:
    from proj.storage.vector_store.chroma_store import ChromaStore

    return ChromaStore


def __getattr__(name: str) -> Any:
    if name == "Chroma":
        return _import_chroma()
    elif name == "PGVector":
        return _import_pgvector()
    else:
        raise AttributeError(f"Could not find: {name}")


__all__ = ["Chroma", "PGVector"]
