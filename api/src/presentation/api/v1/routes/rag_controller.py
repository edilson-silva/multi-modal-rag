from io import BytesIO
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel

from src.infrastructure.rag.rag import RAG

router = APIRouter(prefix='/rag', tags=['rag'])


class SearchRequest(BaseModel):
    owner: str
    query: str


@router.post('/train')
async def train(
    file: Annotated[UploadFile, File(description='PDF file to train the RAG')],
    name: str = Form(...),
):
    if file.content_type != 'application/pdf':
        raise HTTPException(
            status_code=400,
            detail='Invalid file type. Only PDF files are allowed.',
        )

    rag = RAG(name)
    file_content = await file.read()
    bytesio_file = BytesIO(file_content)
    res = rag.train(bytesio_file, filename=file.filename or 'document.pdf')

    return {
        'filename': file.filename,
        'content_type': file.content_type,
        'size': file.size,
        **res,
    }


@router.post('/search')
async def search(body: SearchRequest):
    rag = RAG(body.owner)
    return rag.retrieve(body.query)
