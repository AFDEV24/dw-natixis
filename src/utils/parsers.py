from pypdf import PdfReader
from pathlib import Path
from cohere import RerankResponseResultsItem, Client as CohereClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from semantic_router.encoders import OpenAIEncoder
from semantic_chunkers import StatisticalChunker
from semantic_chunkers.schema import Chunk
from langchain_core.documents import Document
from pathlib import Path

from src import ENV
from src.utils.logger import get_logger
from src.utils.clients import get_cohere_client
from src.models.pinecone import PineconeResults

ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
logger = get_logger(ETC_PATH / "logs")


def parse_pdf(file: Path) -> list[str]:
    """
    Parse a pdf file into a list of strings where each entry in the list is the contents of a page.
    """
    reader = PdfReader(file)
    return [page.extract_text() for page in reader.pages]


def recursively_chunk_content(
    content: Document,
    chunk_size: int = int(ENV["CHUNK_SIZE"]),
    chunk_overlap: int = int(ENV["CHUNK_OVERLAP"]),
) -> list[Document]:
    """
    Splits the content of a given document into smaller chunks based on specified separators and encoding.
    The separators used for splitting include newlines, spaces, punctuation marks, and other special characters.
    Chunk size and overlap are set as env vars.

    Args:
        content (Document): The input document to be split into chunks.
        encoding_name (str): Encoding used for tokenizing. Defaults to cl100k_base as it's used for 3rd gen OpenAI embedding models.

    Returns:
        list[Document]: A list of chunked documents, each representing a portion of the original document content.
    """
    # encoding = tiktoken.encoding_for_model(ENV["CHAT_MODEL"]).name
    encoding = "cl100k_base"
    text_splitter = RecursiveCharacterTextSplitter().from_tiktoken_encoder(
        encoding_name=encoding,
        separators=[
            "\n\n",
            "\n",
            ".",
            ",",
            " ",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.debug(
        f"Chunking using encoding '{encoding}' with separators: {text_splitter._separators}, chunk_size: {text_splitter._chunk_size}, chunk_overlap: {text_splitter._chunk_overlap}"
    )

    documents = text_splitter.split_documents([content])
    return documents


async def statistically_chunk_content(
    contents: list[str], encoding_name: str = ENV["EMBEDDING_MODEL"]
) -> list[tuple[int, str]]:
    """
    Statistically splits the content of a given document into smaller chunks.

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
    client: CohereClient, query: str, results: list[PineconeResults], top_n: int = int(ENV["TOP_RERANK"])
) -> list[PineconeResults]:
    """
    Reranks the results from a vector store based on semantic relavance with respect to the given query.

    Args:
        client: Cohere client used to rerank.
        query (str): User question we are looking to answer.
        results (list[PineconeResults]): The results queried from the vector store along with the metadata.
        top_n (int): Top n chunks to return after reranking. Loaded from .env file.

    Returns:
        list[PineconeResults]: The top n reranked results.
    """
    client = get_cohere_client()
    reranked_results: list[RerankResponseResultsItem] = client.rerank(
        query=query,
        documents=[result.metadata.text for result in results],
        model="rerank-english-v3.0",
        top_n=top_n,
        return_documents=False,
    ).results
    logger.info(f"Reranked chunks -> {[res.index for res in reranked_results]}")
    return [results[res.index] for res in reranked_results]
