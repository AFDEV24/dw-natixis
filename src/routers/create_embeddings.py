import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from langchain_core.documents import Document
from pinecone import Index, Vector

from src.models.requests import CreateEmbeddingsRequest
from src.models.response import CreateEmbeddingsResponse
from src.models.pinecone import PineconeMetadata
from src.utils.embeddings import embed_openai
from src.utils.clients import get_pinecone_index, get_openai_client, OpenAIClient
from src.utils.decorators import async_retry
from src.utils.hashers import hash_string
from src.utils.logger import get_logger
from src.utils.parsers import parse_pdf, statistically_chunk_content

# from src import ENV, DIMENSIONS

ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
router = APIRouter()
logger = get_logger(file_path=ETC_PATH / "logs")

# Ensure tmp download path is created
TMP_PATH: Path = ETC_PATH / "tmp"
TMP_PATH.mkdir(parents=True, exist_ok=True)
PARTIAL_SUFFIX: str = ".part"


@router.post("/create_embeddings")
async def create_embeddings(
    request: CreateEmbeddingsRequest,
) -> CreateEmbeddingsResponse:
    """
    Endpoint to create and store embeddings for a document fetched from a given URL.

    Args:
        request (CreateEmbeddingsRequest): Request object containing the URL of the document to be processed,
        along with client and project information.

    Returns:
        CreateEmbeddingsResponse: Response object containing details of the operation.

    Workflow:
        1. Downloads the file from the specified URL.
        2. Runs a mock OCR extraction on the downloaded file to produce a langchain Document.
        3. Cleans up by deleting the downloaded file from the temporary storage.
        4. Splits the OCR output into smaller chunks.
        5. Adds metadata to each chunk and generates unique IDs for the vectors.
        6. Checks if a Pinecone index exists for the client; if not, creates one for the client.
        7. Uploads the embeddings to the vector database.
    """
    # Get pinecone index, create if it doesnt exist for client
    pc_index = await get_pinecone_index(request.client)
    openai_client = get_openai_client()
    file_ids: dict[str, tuple[list[str], int]] = {}
    # for url in request.urls:

    # TODO: REMOVE HARD CODED PATH FOR FILES WHEN FILE FETCHING IS IMPLEMENTED
    samples_path = ETC_PATH / "sample_files" / "colorado_pension"
    files = [p for p in samples_path.iterdir() if p.is_file()]
    for file in files:
        # Set file path
        # Epoch time - microseconds to minimize race condition when two files with same name are downloaded at the same time
        timestamp: int = int(datetime.now(timezone.utc).timestamp() * 10**6)
        # file_name: str = unquote(str(os.path.basename(urlparse(url).path)))
        file_name = file.stem
        # stamped_file_name = str(timestamp) + "_" + file_name
        # file_path: Path = TMP_PATH / stamped_file_name

        # Download file to disk
        # await _download_file(url, file_path)

        parsed_pdf = parse_pdf(file)

        # Clean up tmp space
        # os.remove(file_path)
        # logger.debug(f"Cleaned up/deleted {file_path}.")

        chunks: list[tuple[int, str]] = await statistically_chunk_content(parsed_pdf)
        logger.info(f"Split content into {len(chunks)} chunks.")

        # Add metadata to documents and generate ids for vectors manually
        ids, documents = _add_metadata(request, chunks, file_name, timestamp)
        file_ids[file_name] = (ids, timestamp)

        # Upload embeddings to vector db
        await _upload_to_pinecone(pc_index, openai_client, documents, ids, request.project)

    return CreateEmbeddingsResponse(
        ids=file_ids,
        details=f"Sucessfully added documents to vector store under index '{request.client}' with namespace '{request.project}'",
    )


# @async_retry(logger, max_attempts=3, initial_delay=1, backoff_base=2)
# async def _download_file(url: str, download_path: Path) -> None:
#     """
#     Downloads a file from the given URL and saves it to disk, supporting resumable downloads.

#     Args:
#         url (str): The URL to download the file from.
#         download_path (str): The path and file name of the file.

#     Raises:
#         HTTPException: If the response status is not successful.
#     """

#     async def _incremental_download(session: ClientSession, url: str, download_path: Path) -> None:
#         partial_file: Path = download_path.with_suffix(PARTIAL_SUFFIX)
#         mode: str = "wb" if not partial_file.exists() else "ab"
#         existing_size: int = partial_file.stat().st_size if partial_file.exists() else 0
#         headers: dict[str, str] = {"Range": f"bytes={existing_size}-"} if existing_size else {}

#         async with session.get(url, headers=headers) as response:
#             response.raise_for_status()

#             # mypy failing to recognize type for mode
#             async with aiofiles.open(str(partial_file), mode) as f:  # type: ignore
#                 async for chunk in response.content.iter_chunked(1024 * 10):  # Reads response in 10KB chunks
#                     await f.write(chunk)
#                     await f.flush()

#         # Remove .part suffix when download is complete
#         partial_file.rename(download_path)

#     async with ClientSession() as session:
#         try:
#             logger.debug(f"Downloading to {download_path}.")

#             start = time.time()
#             await _incremental_download(session, url, download_path)
#             file_size: float = round(os.path.getsize(download_path) / 1024, 2)  # File size in KB

#             logger.info(f"Downloaded file of size {file_size}KB in {round(time.time()-start, 2)}s.")
#         except HTTPException as e:
#             msg = f"Failed to download file from url: {e}"
#             logger.error(msg)
#             raise HTTPException(status_code=500, detail=msg)


# async def _pdf_to_documents(file_path: Path) -> dict[int, Document]:
#     """
#     Mock function to extract OCR content from a PDF file.

#     Args:
#         file_path (Path): Path object to file for which OCR content is to be extracted.

#     Returns:
#         Document: A Document object containing the OCR-extracted text from the specified file.

#     Notes:
#         - This function currently supports only specific sample files.
#         - The OCR content is loaded from corresponding JSON files in the 'ocr_samples' directory.
#     """
#     try:
#         if str(file_path).endswith(".pdf"):
#             contents = parse_pdf(file_path)
#             logger.debug(f"Size of content from {file_path.stem}: {round(sys.getsizeof(contents.values())/1024, 2)}KB")
#         else:
#             raise TypeError(f"Unsupported file type for {file_path.stem}")
#     except MemoryError:
#         logger.error(f"Not enough memory for file of size {round(os.path.getsize(file_path.stem) / 1024 / 1024, 4)}MB")

#     return {idx: Document(page_content=content) for idx, content in contents.items()}


def _add_metadata(
    request: CreateEmbeddingsRequest, chunks: list[tuple[int, str]], file_name: str, timestamp: int
) -> tuple[list[str], list[Document]]:
    """
    Adds metadata to each document in the provided list and generates unique IDs for each document chunk.

    Args:
        documents (list[Document]): List of Document objects to which metadata will be added.
        file_name (str): Name of the file to be included in the metadata of each document.
        timestamp (int): Timestamp for when the endpoint was called.

    Returns:
        tuple[list[str], list[Document]]: A tuple containing two elements:
            - A list of unique string IDs generated for each document chunk.
            - The list of Document objects with updated metadata.
    """
    ids: list[str] = []
    documents: list[Document] = []
    for chunk_number, chunk in enumerate(chunks):
        page_num, text = chunk
        metadata = PineconeMetadata(
            region=request.region,
            fund_name=request.fund_name,
            date=request.date,
            chunk_id=chunk_number,
            name=file_name,
            timestamp=timestamp,
            text=text,
            page=page_num,
        )
        documents.append(Document(page_content=text, metadata=asdict(metadata)))
        ids.append(hash_string(f"{request.date}{request.fund_name}{file_name}{page_num}{chunk_number}"))
    return (ids, documents)


# @async_retry(logger=logger, max_attempts=1, initial_delay=1, backoff_base=2)
async def _upload_to_pinecone(
    index: Index,
    openai_client: OpenAIClient,
    documents: list[Document],
    ids: list[str],
    project: str,
) -> None:
    """
    Internal function to upload documents to a Pinecone index with retries in case of failures.

    Args:
        index (Index): Pinecone index to which the documents will be uploaded.
        client (str): Client/index name associated with the upload.
        project (str): Project name/namespace associated with the upload.
        documents (list[Document]): List of chunked documents to upload.
        ids (list[str]): The list of IDs corresponding to the documents in order.
    """
    # dimensions: int = index.describe_index_stats().dimension
    start = time.time()
    content_vectors = await embed_openai(openai_client, [doc.page_content for doc in documents], 2772)
    # Hardcoding 10% weighting for metadata in embedding
    metadata_vectors = await embed_openai(openai_client, [str(doc.metadata) for doc in documents], 300)
    # Concate vectorspaces
    concate_vectors = [content_vector + metadata_vectors[idx] for idx, content_vector in enumerate(content_vectors)]
    logger.info(f"Took {round(time.time()-start, 2)}s to embed {len(concate_vectors)} documents and metadata.")

    start = time.time()
    vectors = [
        Vector(id=ids[idx], values=concate_vectors[idx], metadata=doc.metadata) for idx, doc in enumerate(documents)
    ]
    index.upsert(vectors, namespace=project)
    logger.info(f"Took {round(time.time()-start, 2)}s to upsert {len(vectors)} documents.")
