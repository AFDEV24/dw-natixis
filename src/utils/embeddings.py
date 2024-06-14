from functools import lru_cache
from pathlib import Path

from openai import Client as OpenAIClient
from langchain_community.embeddings import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore as LCPinecone
from pinecone import Index  # type: ignore
from pydantic.v1.types import SecretStr  # Langchain requires pydantic v1...

from src.utils.logger import get_logger
from src import ENV


ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
logger = get_logger(ETC_PATH / "logs")


@lru_cache(maxsize=8)
def get_lc_pinecone(index: Index, project: str) -> LCPinecone:
    embeddings: BedrockEmbeddings | OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(api_key=SecretStr(ENV["OPENAI_API_KEY"]), model=ENV["EMBEDDING_MODEL"])
    return LCPinecone(index=index, namespace=project, embedding=embeddings)


async def embed_openai(client: OpenAIClient, text: str | list[str], dimensions: int) -> list[list[float]]:
    """
    Returns a list of list of vectors. The outter list is of the same size as input text list.
    """
    embeddings = client.embeddings.create(model=ENV["EMBEDDING_MODEL"], input=text, dimensions=dimensions).data
    return [emb.embedding for emb in embeddings]
