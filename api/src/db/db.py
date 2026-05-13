import os
from typing import List

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector


class DB:
    def __init__(self):
        self.DB_URL = os.environ.get('DB_URL')
        self.connection = PGVector(
            embeddings=OpenAIEmbeddings(),
            connection=self.DB_URL,
            use_jsonb=True,
        )

    def insert(self, collection_name: str, documents: List[Document]):
        """Insert documents into a collection

        Args:
            collection_name (str): Name of collection to store the documents
            documents (List[Document]): List of documents to be stored
        """
        self.connection.add_documents(
            documents=documents,
            collection_name=collection_name,
        )
