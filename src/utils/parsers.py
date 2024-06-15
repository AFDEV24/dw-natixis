from pathlib import Path

from cohere import Client as CohereClient
from cohere import RerankResponseResultsItem
from pypdf import PdfReader
from semantic_chunkers import StatisticalChunker
from semantic_chunkers.schema import Chunk
from semantic_router.encoders import OpenAIEncoder

from src import ENV
from src.models.pinecone import PineconeRecord
from src.utils.clients import get_cohere_client
from src.utils.logger import get_logger

ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
logger = get_logger(ETC_PATH / "logs")


def parse_pdf(file: Path) -> list[str]:
    """
    Parse a pdf file into a list of strings where each entry in the list is the contents of a page.
    """
    reader = PdfReader(file)
    return [page.extract_text() for page in reader.pages]


async def statistically_chunk_content(
    contents: list[str], encoding_name: str = ENV["EMBEDDING_MODEL"]
) -> list[tuple[int, str]]:
    """
    Statistically splits document into smaller chunks to optimize for semantic meaning of each chunk.

    Args:
        content (list[str]): List of strings representing contents of each page.
        encoding_name (str): Encoding used for tokenizing.

    Returns:
        List containing tuple pairs, first element of pair is the page number where the chunk is from, the second
        element is the chunk.
    """
    encoder = OpenAIEncoder(name=encoding_name, openai_api_key=ENV["OPENAI_API_KEY"])
    chunker = StatisticalChunker(
        encoder=encoder,
        enable_statistics=True,
        max_split_tokens=500,
        split_tokens_tolerance=20,
    )
    chunked_contents: list[list[Chunk]] = chunker(docs=contents)
    return [
        (int(page_num), chunk.content) for page_num, page_chunks in enumerate(chunked_contents) for chunk in page_chunks
    ]


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
    client = get_cohere_client()
    documents: list[str] = [str(record.metadata) for record in records]
    logger.debug(f"Documents to rerank: {documents}")
    ranked_records: list[RerankResponseResultsItem] = client.rerank(
        query=query,
        documents=documents,
        model="rerank-english-v3.0",
        top_n=top_n,
        return_documents=False,
    ).results
    logger.info(f"Reranked chunks -> {[record.index for record in ranked_records]}")
    return [records[record.index] for record in ranked_records]
