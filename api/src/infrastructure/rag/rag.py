import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Dict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader, PdfWriter

from src.core.settings import settings
from src.infrastructure.db.db import DB
from src.infrastructure.rag.util.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from src.infrastructure.storage.minio_storage import MinIOStorage


class RAG:
    def __init__(
        self,
        owner: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        """Initialize the RAG with basic config

        Args:
            owner (str): Resources' owner name
            chunk_size (int): The documents chunk size. Defaults to 500.
            chunk_overlap (int): The documents chunk overlap. Defaults to 100.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.owner = owner
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def train(
        self, file: BinaryIO, filename: str = 'document.pdf'
    ) -> Dict[str, Any]:
        """Train the RAG based on loader content

        Args:
            file (BinaryIO): File like to be readed and used to train the rag
            filename (str): Original filename, used as MinIO object prefix
        """
        content = file.read()
        file_stem = Path(filename).stem

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        storage = MinIOStorage()

        # Upload the full PDF
        storage.upload(f'{self.owner}/{filename}', content)

        # Load pages and upload each one as a single-page PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()  # one Document per page

        reader = PdfReader(tmp_path)
        for i, doc in enumerate(documents):
            writer = PdfWriter()
            writer.add_page(reader.pages[i])
            buf = BytesIO()
            writer.write(buf)
            page_object = f'{self.owner}/{file_stem}/page_{i}.pdf'
            storage.upload(page_object, buf.getvalue())
            doc.metadata['page_object'] = page_object

        # Split into chunks — metadata (including page_object) is preserved
        chunks = self.splitter.split_documents(documents)

        db = DB(self.owner)
        db.insert(self.owner, chunks)

        return {'filename': filename, 'page_count': len(documents)}

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve an AI-generated answer grounded in the top-n db chunks

        Args:
            query (str): Text to search for

        Returns:
            Dict with 'answer' (str) and 'sources' (list of presigned URLs)
        """
        db = DB(self.owner)
        docs = db.search(query, 5, threshold=settings.SIMILARITY_THRESHOLD)

        if not docs:
            return {
                'answer': (
                    'Não tenho conhecimentos em minha base para lhe '
                    'fornecer uma resposta.'
                ),
                'sources': [],
            }

        context = '\n\n'.join(
            f'[{i + 1}] {doc.page_content}' for i, doc in enumerate(docs)
        )

        sources = self._build_sources(docs)

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
        return {'answer': response.content, 'sources': sources}

    def _build_sources(self, docs: List) -> List[str]:
        storage = MinIOStorage()
        seen = set()
        urls: list[str] = []
        for doc in docs:
            obj = doc.metadata.get('page_object')
            if obj and obj not in seen:
                seen.add(obj)
                urls.append(storage.presigned_url(self.owner, obj))
        return urls
