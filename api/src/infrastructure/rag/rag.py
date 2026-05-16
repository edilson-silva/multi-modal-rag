import tempfile
from typing import Any, BinaryIO, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.infrastructure.db.db import DB


class RAG:
    def __init__(
        self,
        db_collection_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        """Initialize the RAG with basic config

        Args:
            chunk_size (int): The documents chunk size. Defaults to 500.
            chunk_overlap (int): The documents chunk overlap. Defaults to 100.
            db_collection_name (str): The database collection name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_collection_name = db_collection_name
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def train(self, file: BinaryIO) -> Dict[str, Any]:
        """Train the RAG based on loader content

        Args:
            file (BinaryIO): File like to be readed and used to train the rag
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = file.read()
            tmp.write(content)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        chunks = self.splitter.split_documents(documents)

        db = DB(self.db_collection_name)
        db.insert(self.db_collection_name, chunks)

        return {'filename': tmp.name, 'page_count': len(documents)}
