import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Dict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from pypdf import PdfReader, PdfWriter

from src.core.settings import settings
from src.infrastructure.db.db import DB
from src.infrastructure.rag.util.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from src.infrastructure.storage.minio_storage import MinIOStorage


class RAG:
    _MIN_IMAGE_BYTES = 1024
    _MIN_OCR_CHARS = 20

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

        Returns:
            Dict[str, Any]: The filename and number of generated documents
        """
        content = file.read()
        file_stem = Path(filename).stem

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        storage = MinIOStorage()

        # Load pages and upload each one as a single-page PDF
        loader = PyPDFLoader(tmp_path)
        text_documents: List[Document] = loader.load()  # one Document per page

        reader = PdfReader(tmp_path)
        image_documents: List[Document] = []

        for i, document in enumerate(text_documents):
            writer = PdfWriter()
            writer.add_page(reader.pages[i])
            buf = BytesIO()
            writer.write(buf)
            page_object = f'{self.owner}/{file_stem}/page_{i}.pdf'
            storage.upload(page_object, buf.getvalue())
            document.metadata['page_object'] = page_object

            image_documents.extend(
                self._extract_page_images(
                    reader.pages[i],
                    document.metadata,
                    file_stem,
                    i,
                    storage,
                )
            )

        # Split text chunks — metadata (including page_object) is preserved
        chunks = self.splitter.split_documents(text_documents)

        db = DB(self.owner)
        db.insert(self.owner, chunks + image_documents)

        # Upload the full PDF
        storage.upload(f'{self.owner}/{filename}', content)

        return {'filename': filename, 'page_count': len(text_documents)}

    def _extract_page_images(
        self,
        page: Any,
        page_metadata: Dict,
        file_stem: str,
        page_index: int,
        storage: MinIOStorage,
    ) -> List[Document]:
        """Upload each image from a PDF page to MinIO and OCR it.

        Args:
            page: pypdf object page
            page_metadata: metadata dict from the page Document (copied
                into each image Document)
            file_stem: PDF filename without extension, used as MinIO prefix
            page_index: zero-based page number
            storage: MinIOStorage instance

        Returns:
            List[Document] List of Documents from images' OCR-extracted text
        """
        docs: List[Document] = []

        for i, image in enumerate(page.images):
            if len(image.data) < self._MIN_IMAGE_BYTES:
                continue

            png_bytes = self._get_png_image(image.data)
            ocr_text = self._extract_image_text_content(png_bytes)

            if len(ocr_text.strip()) < self._MIN_OCR_CHARS:
                continue

            img_object = (
                f'{self.owner}/{file_stem}/page_{page_index}_img_{i}.png'
            )
            storage.upload(
                img_object,
                png_bytes,
                content_type='image/png',
            )

            docs.append(
                Document(
                    page_content=ocr_text,
                    metadata={**page_metadata, 'image_object': img_object},
                )
            )
        return docs

    def _get_png_image(self, image_bytes: bytes) -> bytes:
        """Convert image bytes into a png image bytes object

        Args:
            image_bytes (bytes): The image bytets

        Returns:
            bytes: Image converted into png
        """
        buffer = BytesIO()

        with Image.open(BytesIO(image_bytes)) as img:
            img.save(buffer, format='PNG')

        return buffer.getvalue()

    def _extract_image_text_content(self, image_bytes: bytes) -> str:
        """Extract text from an image using a vision LLM.

        Args:
            image_bytes (bytes): The image bytes.

        Returns:
            str: The extracted image text.
        """
        b64 = base64.b64encode(image_bytes).decode()
        llm = ChatOpenRouter(
            model=settings.OCR_MODEL,
            openrouter_api_key=settings.OPENROUTER_KEY,
        )
        response = llm.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/png;base64,{b64}'
                            },
                        },
                        {
                            'type': 'text',
                            'text': (
                                'Extract all visible text in this image. '
                                'Return only the extracted text, nothing else. '
                                'If there is no text, return an empty string.'
                            ),
                        },
                    ]
                )
            ]
        )
        return response.content

    def _build_sources(self, docs: List) -> List[str]:
        """Get the list of documents' sources as presigned urls

        Args:
            docs (List[Document]): The list of documents to extract source

        Returns:
            List[str]: The list of found documents sources
        """
        storage = MinIOStorage()
        seen = set()
        urls: List[str] = []

        for doc in docs:
            for key in ('page_object', 'image_object'):
                obj = doc.metadata.get(key)

                if obj and obj not in seen:
                    seen.add(obj)
                    urls.append(storage.presigned_url(self.owner, obj))

        return urls

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve an AI-generated answer grounded in the top-n db chunks

        Args:
            query (str): Text to search for

        Returns:
            Dict[str, Any]: The answer and the list of pressigned urls sources
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
