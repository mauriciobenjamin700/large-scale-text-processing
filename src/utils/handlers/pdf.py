from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFHandler:
    """
    Handler for loading and processing PDF documents.

    Methods:
        load_pdf: Load a PDF file and return its documents.
        split_documents: Split documents into smaller chunks.
    """
    @staticmethod
    def load_pdf(file_path: str) -> list[Document]:
        """
        Load a PDF file and return its documents.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            list[Document]: A list of Document objects extracted from the PDF.
        """
        loader = PyPDFLoader(file_path)
        return loader.load()

    @staticmethod
    def split_documents(
        documents: list[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> list[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents (list[Document]): The list of Document objects to split.
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap between chunks.

        Returns:
            list[Document]: A list of split Document objects.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        return text_splitter.split_documents(documents)


__all__ = ["PDFHandler"]
