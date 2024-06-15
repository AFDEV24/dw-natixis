import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from pinecone import Index, Vector

from src import DIMENSIONS, ENV
from src.models.pinecone import PineconeMetadata, PineconeRecord
from src.models.requests import CreateEmbeddingsRequest
from src.models.response import CreateEmbeddingsResponse
from src.utils.clients import OpenAIClient, get_openai_client, get_pinecone_index
from src.utils.decorators import async_retry
from src.utils.embeddings import embed_openai
from src.utils.hashers import hash_string
from src.utils.logger import get_logger
from src.utils.parsers import parse_pdf, statistically_chunk_content

ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
router = APIRouter()
logger = get_logger(file_path=ETC_PATH / "logs")


@router.post("/create_embeddings")
async def create_embeddings(request: CreateEmbeddingsRequest) -> CreateEmbeddingsResponse:
    """
    Endpoint to create embeddings from pdf files and upload them to Pinecone.

    Args:
        request (CreateEmbeddingsRequest): Request object containing client, fund name, date, and other metadata.

    Returns:
        CreateEmbeddingsResponse: The response containing the IDs of the uploaded documents and a success message.
    """
    openai_client = get_openai_client()
    pc_index = await get_pinecone_index(request.client)

    # TODO: REMOVE HARD CODED PATH FOR FILES WHEN FILE FETCHING IS IMPLEMENTED
    samples_path = ETC_PATH / "sample_files" / "colorado_pension"
    files = [p for p in samples_path.iterdir() if p.is_file()]
    file_ids: dict[str, tuple[list[str], int]] = {}
    for file in files:
        # Epoch time - microseconds to minimize race condition when two files with same name are downloaded at the same time
        timestamp: int = int(datetime.now(timezone.utc).timestamp() * 10**6)
        file_name: str = file.stem
        parsed_pdf: list[str] = parse_pdf(file)

        chunks: list[tuple[int, str]] = await statistically_chunk_content(parsed_pdf)
        logger.info(f"Split content into {len(chunks)} chunks.")

        records = _create_records(request, chunks, file_name, timestamp)
        vectors = await _create_vectors(openai_client, records)
        await _upload_to_pinecone(pc_index, vectors, request.project)

        file_ids[file_name] = ([record.id for record in records], timestamp)

    return CreateEmbeddingsResponse(
        ids=file_ids,
        details=f"Sucessfully added documents to vector store under index '{request.client}' with namespace '{request.project}'",
    )


def _create_records(
    request: CreateEmbeddingsRequest, chunks: list[tuple[int, str]], file_name: str, timestamp: int
) -> list[PineconeRecord]:
    """
    Create PineconeRecord instances for each text chunk of the file.

    Args:
        request (CreateEmbeddingsRequest): Request object containing client, fund name, date, and other metadata.
        chunks (list[tuple[int, str]]): List of tuples, each containing a page number and a text chunk.
        file_name (str): Name of the file being processed.
        timestamp (int): Timestamp for the document.

    Returns:
        List[PineconeRecord]: List of PineconeRecord instances with metadata and IDs populated.
    """
    records: list[PineconeRecord] = []
    for chunk_number, chunk in enumerate(chunks):
        page_num, text = chunk
        records.append(
            PineconeRecord(
                id=hash_string(f"{request.date}{request.fund_name}{file_name}{page_num}{chunk_number}"),
                metadata=PineconeMetadata(
                    region=request.region,
                    fund_name=request.fund_name,
                    date=request.date,
                    chunk_id=chunk_number,
                    file_name=file_name,
                    timestamp=timestamp,
                    text=text,
                    page=page_num,
                ),
            )
        )
    return records


@async_retry(logger=logger, max_attempts=2, initial_delay=1, backoff_base=2)
async def _create_vectors(openai_client: OpenAIClient, records: list[PineconeRecord]) -> list[Vector]:
    """
    Create vector space for each record by embedding their content and metadata using OpenAI embedding model
    then concatenate the text vectors for a PineconeRecord with its respective metadata vectors.

    The dimension for metadata is hard coded to 300. The dimension for the content of each record is the
    dimension/vector space for the embedding model - 300.

    Each list of floats (a vector) maps to a PineconeRecord.

    Args:
        openai_client (OpenAIClient): OpenAI client used to make embedding model API call.
        records (list[PineconeRecord]): List of PineconeRecord instances to be embedded.

    Returns:
        list[Vector]: List of vectors containing the embeddings and metadata.
    """
    start = time.time()
    metadata_vectors: list[list[float]] = await embed_openai(
        openai_client,
        [record.metadata.to_string_except_text() for record in records],
        300,
    )
    content_vectors: list[list[float]] = await embed_openai(
        openai_client,
        [record.metadata.text for record in records],
        DIMENSIONS[ENV["EMBEDDING_MODEL"]] - 300,
    )
    logger.info(f"Took {round(time.time()-start, 2)}s to embed {len(content_vectors)} text chunks and metadata.")

    concate_vectors = [content_vector + metadata_vectors[idx] for idx, content_vector in enumerate(content_vectors)]
    return [
        Vector(
            id=records[idx].id,
            values=vector,
            metadata=asdict(records[idx].metadata),
        )
        for idx, vector in enumerate(concate_vectors)
    ]


@async_retry(logger=logger, max_attempts=2, initial_delay=1, backoff_base=2)
async def _upload_to_pinecone(index: Index, vectors: list[Vector], project: str) -> None:
    """
    Upload vectors to the Pinecone index.

    Args:
        index (Index): The Pinecone index to upload vectors to.
        vectors (List[Vector]): A list of vectors to be uploaded.
        project (str): The namespace/project in the Pinecone index where the vectors will be stored.

    Returns:
        None
    """
    start = time.time()
    index.upsert(vectors, namespace=project)
    logger.info(f"Took {round(time.time()-start, 2)}s to upsert {len(vectors)} entries to Index '{project}'.")
