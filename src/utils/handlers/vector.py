from typing import Any
from typing_extensions import Literal
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


class VectorHandler:

    @staticmethod
    def map_text_to_vector(
        text: str,
        embeddings: OllamaEmbeddings
    ) -> list[float]:
        """
        Generate an embedding vector for a given text.

        Args:
            text (str): The text to embed.
            embeddings (OllamaEmbeddings): The OllamaEmbeddings instance
                to use.
        Returns:
            list[float]: The embedding vector for the document.
        """
        return embeddings.embed_query(text)

    @staticmethod
    def map_document_to_vector(
        document: Document,
        embeddings: OllamaEmbeddings
    ) -> list[float]:
        """
        Generate an embedding vector for a given document.

        Args:
            document (Document): The Document object to embed.
            embeddings (OllamaEmbeddings): The OllamaEmbeddings instance
                to use.
        Returns:
            list[float]: The embedding vector for the document.
        """
        return embeddings.embed_query(document.page_content)

    @staticmethod
    def create_vector_store(
        embedding_function: OllamaEmbeddings,
        persist_directory: str = "./chroma_langchain_db",
        collection_name: str = "default_collection"
    ) -> Chroma:
        """
        Create a Chroma vector store.

        Args:
            embedding_function (OllamaEmbeddings): The embedding function
                to use.
            persist_directory (str): Directory to persist the vector store.
            collection_name (str): Name of the collection.

        Returns:
            Chroma: The created Chroma vector store.
        """
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
        )

    @staticmethod
    def save_documents_on_vector_store(
        vector_store: Chroma,
        documents: list[Document],
        ids: list[str]
    ) -> None:
        """
        Save documents to the vector store.

        Args:
            vector_store (Chroma): The Chroma vector store.
            documents (list[Document]): List of Document objects to save.
            ids (list[str]): List of IDs corresponding to the documents.
        """
        vector_store.add_documents(documents=documents, ids=ids)

    @staticmethod
    def update_document_on_vector_store(
        vector_store: Chroma,
        document_id: str,
        document: Document
    ) -> None:
        """
        Update a document in the vector store.

        Args:
            vector_store (Chroma): The Chroma vector store.
            document_id (str): The ID of the document to update.
            document (Document): The updated Document object.
        """
        vector_store.update_document(
            document_id=document_id,
            document=document
        )

    @staticmethod
    def delete_documents_from_vector_store(
        vector_store: Chroma,
        ids: list[str]
    ) -> None:
        """
        Delete documents from the vector store.

        Args:
            vector_store (Chroma): The Chroma vector store.
            ids (list[str]): List of IDs of the documents to delete.
        """
        vector_store.delete(ids=ids)

    @staticmethod
    def find_documents_by_similarity(
        vector_store: Chroma,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None
    ) -> list[Document]:
        """
        Find documents similar to the query.

        Args:
            vector_store (Chroma): The Chroma vector store.
            query (str): The query string.
            k (int): Number of similar documents to retrieve.
            filter (dict[str, str] | None): Optional filter for the search.

        Returns:
            list[Document]: List of similar Document objects.
        """
        return vector_store.similarity_search(query, k=k, filter=filter)

    @staticmethod
    def find_documents_by_vector(
        vector_store: Chroma,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None
    ) -> list[Document]:
        """
        Find documents similar to the given embedding vector.

        Args:
            vector_store (Chroma): The Chroma vector store.
            embedding (list[float]): The embedding vector to search with.
            k (int): Number of similar documents to retrieve.
            filter (dict[str, str] | None): Optional filter for the search.

        Returns:
            list[Document]: List of similar Document objects.
        """
        return vector_store.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter
        )

    @staticmethod
    def find_by_query_on_retriever(
        retriever: VectorStoreRetriever,
        query: str,
        filter: dict[str, str] | None = None
    ) -> list[Document]:
        """
        Find documents using the retriever based on the query.

        Args:
            retriever (VectorStoreRetriever): The retriever object.
            query (str): The query string.

        Returns:
            list[Document]: List of retrieved Document objects.
        """
        return retriever.invoke(query, filter=filter)

    @staticmethod
    def map_to_retriever(
        vector_store: Chroma,
        search_type: Literal["similarity", "mmr"] = "mmr",
        search_kwargs: dict[str, Any] = {"k": 1}
    ) -> VectorStoreRetriever:
        """
        Map the vector store to a retriever.

        Args:
            vector_store (Chroma): The Chroma vector store.
            search_type (str): The type of search to perform.
            search_kwargs (dict): Additional keyword arguments for the search.

        Returns:
            Retriever: The retriever object.
        """
        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )


__all__ = ["VectorHandler"]
