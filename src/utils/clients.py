import boto3
from pathlib import Path
from functools import lru_cache
from async_lru import alru_cache
import boto3.session
from openai import Client as OpenAIClient
from cohere import Client as CohereClient
from minio import Minio
from pinecone import Index, Pinecone, ServerlessSpec  # type: ignore
from typing import Any

from src import ENV, DIMENSIONS
from src.utils.logger import get_logger

ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
logger = get_logger(ETC_PATH / "logs")


@lru_cache(maxsize=16)
def get_cohere_client() -> CohereClient:
    return CohereClient(api_key=ENV["COHERE_API_KEY"])


@lru_cache(maxsize=16)
def get_openai_client() -> OpenAIClient:
    return OpenAIClient(api_key=ENV["OPENAI_API_KEY"])


@lru_cache(maxsize=32)
def get_bedrock_client() -> Any:  # tmp typing
    return boto3.client(
        "bedrock-runtime",
        "us-east-1",
        aws_access_key_id=ENV["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=ENV["AWS_SECRET_ACCESS_KEY"],
    )


@lru_cache(maxsize=4)
def get_minio_client() -> Minio:
    """
    Dependency function to configure and serve Minio client. Client is cached for resuse.

    Can cache multiple Minio clients in case different customers/projects have different buckets.
    """
    return Minio(
        endpoint=f"{ENV['MINIO_HOSTNAME']}:{ENV['MINIO_API_PORT']}",
        access_key=ENV["MINIO_ROOT_USER"],
        secret_key=ENV["MINIO_ROOT_PASSWORD"],
        secure=False,  # since its localhost, no https
    )


@lru_cache(maxsize=2)
def get_pinecone_client() -> Pinecone:
    """
    Configure and serve Pinecone client. Client is cached for reuse.
    """
    return Pinecone(api_key=ENV["PINECONE_API_KEY"])


@alru_cache(maxsize=8)  # Can increase cache size if we have more indexes
async def get_pinecone_index(
    client: str, dimension: int | None = None, metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1"
) -> Index:
    """
    Creates a Pinecone index if it doesn't already exist.
    NOTE: We are only creating an index here for demo/testing purposes!!

    Args:
        client (str): Name of the index.
        dimension (int | None): Dimensionality of the vectors to be indexed. Defaults to None.
        metric (str): Distance metric to use. Defaults to "cosine".
        cloud (str): Cloud provider for infra. Defaults to "aws".
        region (str): Region where pod will be located. Defaults to "us-east-1".

    Returns:
        Pinecone Index for the client, which represents a connection.
    """
    pinecone_client = get_pinecone_client()
    existing_index_names = [idx.name for idx in pinecone_client.list_indexes()]
    logger.debug(f"Index list {existing_index_names}")  # NOTE: might not see the logs due to cache

    # NOTE: We are only creating an index here for demo/testing purposes
    # Real endpoint should not create an index if it doesnt exist, it should throw an error
    if client not in existing_index_names:
        pinecone_client.create_index(
            name=client,
            dimension=DIMENSIONS[ENV["EMBEDDING_MODEL"]] if dimension is None else dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        logger.info(f"Created index for client '{client}'.")
    else:
        logger.debug(f"Index exists for client '{client}'.")

    return pinecone_client.Index(name=client)
