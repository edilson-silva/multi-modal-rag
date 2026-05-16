from fastapi import APIRouter

from src.presentation.api.v1.routes import rag_controller

api_router = APIRouter()
api_router.include_router(rag_controller.router)
