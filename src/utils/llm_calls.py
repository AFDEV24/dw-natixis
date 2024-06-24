from pathlib import Path
from cohere import Client as CohereClient
from cohere import RerankResponseResultsItem
from openai.types.chat import ChatCompletionMessageParam

from src import ENV
from src.models.pinecone import PineconeRecord
from src.utils.connections import get_cohere_client, OpenAIClient
from src.utils.decorators import async_retry
from src.utils.logger import get_logger


ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
logger = get_logger(ETC_PATH / "logs")


@async_retry(logger, max_attempts=3, initial_delay=1, backoff_base=2)
async def rerank(
    client: CohereClient, query: str, records: list[PineconeRecord], top_n: int = int(ENV["TOP_RERANK"])
) -> list[PineconeRecord]:
    """
    Reranks the records from a vector store based on semantic relavance with respect to the given query.

    Args:
        client: Cohere client used to rerank.
        query (str): User question we are looking to answer.
        records (list[PineconeRecord]): The records queried from the vector store along with the metadata.
        top_n (int): Top n chunks to return after reranking. Loaded from .env file.

    Returns:
        list[PineconeRecord]: The top n reranked results.
    """
    documents: list[str] = [str(record.metadata) for record in records]  # record metadata includes chunk text
    if ENV["DEBUG"] == "TRUE":
        for idx, doc in enumerate(documents):
            logger.debug(f"Chunk {idx} to rerank: {doc}\n----------------\n")
    ranked_records: list[RerankResponseResultsItem] = client.rerank(
        query=query,
        documents=documents,
        model=ENV["RERANK_MODEL"],
        top_n=top_n,
        return_documents=False,
    ).results
    logger.info(f"Reranked chunks -> {[record.index for record in ranked_records]}")
    return [records[record.index] for record in ranked_records]


@async_retry(logger, max_attempts=3, initial_delay=1, backoff_base=2)
async def openai_chat(
    openai_client: OpenAIClient,
    model: str,
    prompt: list[ChatCompletionMessageParam],
    temperature: float,
    default_idk: str = "I dont know.",
) -> str:
    answer = openai_client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=temperature,
    )
    return answer.choices[0].message.content if answer.choices[0].message.content else default_idk
