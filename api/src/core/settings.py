from dotenv import find_dotenv, load_dotenv
from pydantic import Field, PostgresDsn
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
    DATABASE_URL: PostgresDsn = Field(..., env='DATABASE_URL')


settings = Settings()
