from pathlib import Path
from pypdf import PdfReader
from semantic_chunkers import StatisticalChunker
from semantic_chunkers.schema import Chunk
from semantic_router.encoders import OpenAIEncoder

from src import ENV

ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"


def parse_pdf(file: Path) -> list[str]:
    """
    Parse a pdf file into a list of strings where each entry in the list is the contents of a page.
    """
    reader = PdfReader(file)
    return [page.extract_text() for page in reader.pages]


async def statistically_chunk_content(
    pages: list[str], encoding_name: str = ENV["EMBEDDING_MODEL"]
) -> list[tuple[int, str]]:
    """
    Statistically splits document into smaller chunks to optimize for semantic meaning of each chunk.

    Args:
        content (list[str]): List of strings representing contents of each page.
        encoding_name (str): Encoding used for tokenizing.

    Returns:
        List containing tuple pairs, first element of pair is the page number (index starting at 1)
        where the chunk is from, the second element is the text chunk.
    """
    encoder = OpenAIEncoder(name=encoding_name, openai_api_key=ENV["OPENAI_API_KEY"])
    chunker = StatisticalChunker(
        encoder=encoder,
        enable_statistics=True,
        max_split_tokens=500,
        split_tokens_tolerance=20,
    )
    chunked_pages: list[list[Chunk]] = chunker(docs=pages)
    return [
        (int(page_num) + 1, chunk.content)
        for page_num, page_chunks in enumerate(chunked_pages)
        for chunk in page_chunks
    ]
