from langchain_core.document_loaders.base import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.db.db import DB


class RAG:
    def __init__(
        self,
        loader: BaseLoader,
        db_collection_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        """Initialize the RAG with basic config

        Args:
            loader (BaseLoader): The loader object instance
            chunk_size (int): The documents chunk size. Defaults to 500.
            chunk_overlap (int): The documents chunk overlap. Defaults to 100.
            db_collection_name (str): The database collection name
        """
        self.loader = loader
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_collection_name = db_collection_name
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.db = DB()

    def train(self):
        """Train the RAG based on loader content"""
        documents = self.loader.load()
        chunks = self.splitter.split_documents(documents)

        self.db.insert(self.db_collection_name, chunks)
