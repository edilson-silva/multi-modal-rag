from langchain_core.embeddings import Embeddings
from openai import OpenAI

from src.core.settings import settings


class OpenRouterEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.client = OpenAI(
            api_key=settings.OPENROUTER_KEY,
            base_url=settings.OPENROUTER_BASE,
        )
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []

        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )

        return response.data[0].embedding
