from io import BytesIO
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from src.infrastructure.rag.rag import RAG

router = APIRouter(prefix='/rag', tags=['rag'])


@router.post('/train')
async def train(
    file: Annotated[UploadFile, File(description='PDF file to train the RAG')],
    name: str = Form(...),
):  # Validate the file type
    if file.content_type != 'application/pdf':
        raise HTTPException(
            status_code=400,
            detail='Invalid file type. Only PDF files are allowed.',
        )

    rag = RAG(name)
    file_content = await file.read()
    bytesio_file = BytesIO(file_content)
    res = rag.train(bytesio_file)

    return {
        'filename': file.filename,
        'content_type': file.content_type,
        'size': file.size,
        **res,
    }
