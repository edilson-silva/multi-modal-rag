from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from src.core.settings import settings


class DB:
    __instance = None

    def __new__(cls):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)

        return cls.__instance

    def __init__(self, embeddings: Embeddings):
        """Initialize DB connection with basic settings

        Args:
            embeddings (Embeddings): The embeddings model to be used by database
        """
        self.__connection = PGVector(
            embeddings=embeddings,
            connection=settings.DATABASE_URL,
            use_jsonb=True,
        )

    def insert(self, collection_name: str, documents: List[Document]):
        """Insert documents into a collection

        Args:
            collection_name (str): Name of collection to store the documents
            documents (List[Document]): List of documents to be stored
        """
        self.__connection.add_documents(
            documents=documents,
            collection_name=collection_name,
        )
