"""Local embeddings via FastEmbed (ONNX), wrapped in LangChain's Embeddings interface. No API key needed."""

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = MODEL_NAME):
        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # query_embed applies the query-side prefix the BGE models were trained with.
        return next(iter(self._model.query_embed(text))).tolist()
