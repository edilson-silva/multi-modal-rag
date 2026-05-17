from typing import Dict, List
from uuid import uuid4

from langchain_core.documents import Document
from langchain_postgres import PGVector

from src.core.settings import settings
from src.infrastructure.rag.embeddings.openrouter_embeddings import (
    OpenRouterEmbeddings,
)


class DB:
    __instance = None
    __connections: Dict[str, PGVector] = {}

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance

    def __init__(self, collection_name):
        """
        Initialize DB connection with basic settings

        Args:
            collection_name (str): Name of collection to store documents
        """
        self.collection_name = collection_name

        if collection_name not in self.__connections:
            self.__connections[collection_name] = PGVector(
                embeddings=OpenRouterEmbeddings(settings.EMBEDDINGS_MODEL),
                collection_name=collection_name,
                connection=settings.DATABASE_URL,
                use_jsonb=True,
            )

    @property
    def connection(self) -> PGVector:
        return self.__connections[self.collection_name]

    def insert(self, collection_name: str, chunks: List[Document]):
        """Insert documents into a collection

        Args:
            collection_name (str): Name of collection to store the documents
            chunks (List[Document]): List of documents chunks to be stored
        """
        ids = [str(uuid4()) for _ in range(len(chunks))]

        self.__connections[collection_name].add_documents(
            documents=chunks,
            ids=ids,
        )

    def search(self, query: str, n: int = 5) -> List[Document]:
        """Retrieve n most similar documents for a given query text

        Args:
            query (str): Text to search for
            n (int): Number of results to return

        Returns:
            List[Document]: List of similar documents
        """
        return self.connection.similarity_search(query, k=n)
