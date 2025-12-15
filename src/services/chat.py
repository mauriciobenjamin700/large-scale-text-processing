from os.path import exists
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from src.schemas import ChatSessionSchema, DocumentSchema
from src.utils.handlers import PDFHandler, VectorHandler


class ChatService:
    """
    Service class to handle chat sessions, including message processing
    and document management.

    Attributes:
        db (Chroma): The Chroma vector store instance.
        model (ChatOllama): The ChatOllama model instance.
        session (ChatSessionSchema): The current chat session schema.

    Methods:
        send_message(message: str) -> str:
            Process a user message and generate a response.
        add_pdf(pdf_path: str) -> None:
            Add a PDF document to the chat session.
        remove_document(document_id: str) -> None:
            Remove a document from the chat session by its ID.
        remove_pdf(pdf_path: str) -> None:
            Remove a PDF document and its associated data from the
                chat session.
        clear_session() -> None:
            Clear all documents and data from the current chat session.
    """
    def __init__(self, db: Chroma, model: ChatOllama):
        self.db = db
        self.model = model
        self.session: ChatSessionSchema = ChatSessionSchema(
            session_id=str(uuid4()),
            documents=[]
        )

    def send_message(self, message: str) -> str:
        """
        Process a user message and generate a response.

        Args:
            message (str): The user's message.
        Returns:
            str: The generated response.
        """        # Placeholder for message processing logic
        retriever = VectorHandler.map_to_retriever(
            self.db, "similarity", search_kwargs={"k": 5}
        )
        documents = VectorHandler.find_by_query_on_retriever(
            retriever,
            query=message,
        )

        print(f"Found {len(documents)} similar documents for the query.")

        context = "\n".join([doc.page_content for doc in documents])

        print(f"Context prepared for the model. {context}")

        input = [
            {
                "role": "system",
                "content": f"Use this context on the documents:\n{context}"
            },
            {"role": "user", "content": f"Answer: {message}"}
        ]

        response = self.model.invoke(input=input)

        print(f"Model response received: {response}")

        return str(response.content)

    def add_pdf(self, pdf_path: str) -> None:
        """
        Add a PDF document to the chat session.

        Args:
            pdf_path (str): The path to the PDF file.

        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
        """
        if not exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")

        documents = PDFHandler.load_pdf(pdf_path)
        docs = documents
        # filter out empty chunks (some PDFs/pages may produce empty text)
        non_empty_docs = [d for d in docs if (d.page_content or "").strip()]
        empty_count = len(docs) - len(non_empty_docs)
        if empty_count:
            print(
                f"Warning: {empty_count} "
                "empty chunk(s) were dropped before indexing."
                )

        if not non_empty_docs:
            raise ValueError("No text extracted from PDF; aborting load.")

        print(
            f"Split into {len(non_empty_docs)} "
            f"non-empty chunks (dropped {empty_count})."
        )
        ids = [str(uuid4()) for _ in non_empty_docs]

        try:
            VectorHandler.save_documents_on_vector_store(
                vector_store=self.db,
                documents=non_empty_docs,
                ids=ids,
            )
        except Exception as e:
            # provide more context when embeddings/vector store calls fail
            print(f"Error saving documents to vector store: {e}")
            raise
        self.session.documents.append(
            DocumentSchema(
                filename=pdf_path,
                documents_id=ids,
                documents=non_empty_docs,
            )
        )

    def remove_document(self, document_id: str) -> None:
        """
        Remove a document from the chat session by its ID.

        Args:
            document_id (str): The ID of the document to remove.
        Raises:
            ValueError: If the document ID is not found in the session.
        """
        for doc_schema in self.session.documents:
            if document_id in doc_schema.documents_id:
                VectorHandler.delete_documents_from_vector_store(
                    vector_store=self.db,
                    ids=[document_id]
                )
                idx = doc_schema.documents_id.index(document_id)
                del doc_schema.documents_id[idx]
                del doc_schema.documents[idx]
                break

    def remove_pdf(self, pdf_path: str) -> None:
        """
        Remove a PDF document and its associated data from the chat session.

        Args:
            pdf_path (str): The path to the PDF file to remove.
        Raises:
            ValueError: If the PDF file is not found in the session.
        """
        for doc_schema in self.session.documents:
            if doc_schema.filename == pdf_path:
                VectorHandler.delete_documents_from_vector_store(
                    vector_store=self.db,
                    ids=doc_schema.documents_id
                )
                self.session.documents.remove(doc_schema)
                break

    def clear_session(self) -> None:
        """
        Clear all documents and data from the current chat session.
        """
        for doc_schema in self.session.documents:
            VectorHandler.delete_documents_from_vector_store(
                vector_store=self.db,
                ids=doc_schema.documents_id
            )
        self.session.documents.clear()

    def list_pdf(self) -> list[str]:
        """
        List all loaded PDF documents in the current chat session.

        Returns:
            list: A list of filenames of the loaded PDF documents.
        """
        return [doc.filename for doc in self.session.documents]
