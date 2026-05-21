from dotenv import find_dotenv, load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(find_dotenv())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
    )

    APP_NAME: str = 'Multi Modal RAG'
    DATABASE_URL: str = Field(..., env='DATABASE_URL')
    OPENROUTER_BASE: str = Field(..., env='OPENROUTER_BASE')
    OPENROUTER_KEY: str = Field(..., env='OPENROUTER_KEY')
    EMBEDDINGS_MODEL: str = Field(..., env='EMBEDDINGS_MODEL')
    CHAT_MODEL: str = Field(..., env='CHAT_MODEL')
    SIMILARITY_THRESHOLD: float = Field(0.4, env='SIMILARITY_THRESHOLD')
    MINIO_ENDPOINT: str = Field(..., env='MINIO_ENDPOINT')
    MINIO_ACCESS_KEY: str = Field(..., env='MINIO_ACCESS_KEY')
    MINIO_SECRET_KEY: str = Field(..., env='MINIO_SECRET_KEY')
    MINIO_SECURE: bool = Field(False, env='MINIO_SECURE')
    MINIO_BUCKET: str = Field('rag-documents', env='MINIO_BUCKET')
    MINIO_PUBLIC_ENDPOINT: str = Field(..., env='MINIO_PUBLIC_ENDPOINT')
    MINIO_REGION: str = Field('us-east-1', env='MINIO_REGION')
    OCR_MODEL: str = Field(..., env='OCR_MODEL')


settings = Settings()
