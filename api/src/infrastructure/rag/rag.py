import tempfile
from typing import Any, BinaryIO, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.settings import settings
from src.infrastructure.db.db import DB
from src.infrastructure.rag.util.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)


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

    def retrieve(self, query: str) -> str:
        """Retrieve an AI-generated answer grounded in the top-n db chunks

        Args:
            query (str): Text to search for

        Returns:
            str: AI-generated answer based solely on retrieved context
        """
        db = DB(self.db_collection_name)
        docs = db.search(query, 5)

        context = '\n\n'.join(
            f'[{i + 1}] {doc.page_content}' for i, doc in enumerate(docs)
        )

        llm = ChatOpenRouter(
            model=settings.CHAT_MODEL,
            openrouter_api_key=settings.OPENROUTER_KEY,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=USER_PROMPT_TEMPLATE.format(
                    context=context, query=query
                )
            ),
        ]

        response = llm.invoke(messages)
        return response.content
