from functools import lru_cache
from async_lru import alru_cache
from openai import Client as OpenAIClient
from cohere import Client as CohereClient
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


@lru_cache(maxsize=4)
def get_pinecone_client(threads: int = 30) -> Pinecone:
    """
    Get a cached instance of the Pinecone client.

    Args:
        threads (int): The number of threads to use in the Pinecone client . Defaults to 30.

    Returns:
        Pinecone: An instance of the Pinecone client initialized with the API key from the environment.
    """
    return Pinecone(api_key=ENV["PINECONE_API_KEY"], pool_threads=threads)


@alru_cache(maxsize=8)  # Can increase cache size if we have more indexes
async def get_pinecone_index(index_name: str, threads: int = 30) -> Index:
    """
    Get a cached instance of the Pinecone index.

    Args:
        index_name (str): The name of the Pinecone index.
        threads (int): The number of threads to use in the Pinecone client . Defaults to 30.

    Returns:
        Index: An instance of the Pinecone index for the specified client.
    """
    pinecone_client = get_pinecone_client(threads)
    return pinecone_client.Index(name=index_name)
