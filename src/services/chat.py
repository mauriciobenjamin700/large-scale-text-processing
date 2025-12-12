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
        documents = VectorHandler.find_documents_by_similarity(
            self.db,
            query=message,
            k=5
        )

        context = "\n".join([doc.page_content for doc in documents])

        prompt = (
            f"Based on the documents:\n{context}\n\n"
            f"Answer: {message}")

        response = self.model.invoke(prompt)

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
        split_docs = PDFHandler.split_documents(documents)
        ids = [str(uuid4()) for _ in split_docs]

        VectorHandler.save_documents_on_vector_store(
            vector_store=self.db,
            documents=split_docs,
            ids=ids
        )
        self.session.documents.append(
            DocumentSchema(
                filename=pdf_path,
                documents_id=ids,
                documents=split_docs
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
