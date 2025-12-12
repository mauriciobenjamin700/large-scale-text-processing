from langchain_core.documents import Document

from src.core import BaseSchema


class DocumentSchema(BaseSchema):
    """
    Schema representing a document in the chat system.

    Attributes:
        filename (str): The name of the document file.
        documents_id (list[str]): List of document IDs associated with
            the document.
        documents (list[Document]): List of Document objects.
    """
    filename: str
    documents_id: list[str]
    documents: list[Document]


class ChatSessionSchema(BaseSchema):
    """
    Schema representing a chat session.

    Attributes:
        session_id (str): Unique identifier for the chat session.
        documents (list[DocumentSchema]): List of DocumentSchema objects
            associated with the chat session.
    """
    session_id: str
    documents: list[DocumentSchema] = []


__all__ = ["DocumentSchema", "ChatSessionSchema"]
