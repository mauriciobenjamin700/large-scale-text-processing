from langchain_chroma import Chroma

from src.core import embeddings, settings


vector_db: Chroma = Chroma(
    collection_name=settings.CHROMA_COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
)

__all__ = ["vector_db"]
