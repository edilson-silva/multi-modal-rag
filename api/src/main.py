from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse

from src.core.settings import settings

app = FastAPI(
    title=settings.APP_NAME,
    version='1.0.0',
    root_path='/api/v1',
)


@app.api_route('/', methods=['GET', 'POST'])
async def root():
    return RedirectResponse(
        url='/docs', status_code=status.HTTP_307_TEMPORARY_REDIRECT
    )
