from functools import lru_cache

from async_lru import alru_cache
from cohere import Client as CohereClient
from openai import Client as OpenAIClient
from pinecone import Index, Pinecone

from src import ENV


@lru_cache(maxsize=16)
def get_cohere_client() -> CohereClient:
    """
    Get a cached instance of the CohereClient.
    """
    return CohereClient(api_key=ENV["COHERE_API_KEY"])


@lru_cache(maxsize=16)
def get_openai_client() -> OpenAIClient:
    """
    Get a cached instance of the OpenAIClient.
    """
    return OpenAIClient(api_key=ENV["OPENAI_API_KEY"])


@lru_cache(maxsize=1)
def get_pinecone_client(threads: int = 30) -> Pinecone:
    """
    Get a cached instance of the Pinecone client.

    Args:
        threads (int): The number of threads to use in the Pinecone client . Defaults to 30.

    Returns:
        Pinecone: An instance of the Pinecone client initialized with the API key from the environment.
    """
    return Pinecone(api_key=ENV["PINECONE_API_KEY"], pool_threads=threads)
