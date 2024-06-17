import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from openai import AuthenticationError
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from pinecone.exceptions import UnauthorizedException

from src import DIMENSIONS, ENV
from src.models.pinecone import PineconeMetadata, PineconeRecord
from src.models.requests import QueryRequest
from src.models.response import QueryResponse
from src.utils.clients import get_cohere_client, get_openai_client, get_pinecone_index
from src.utils.embeddings import embed_openai
from src.utils.logger import get_logger
from src.utils.parsers import rerank

router = APIRouter()
ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
logger = get_logger(file_path=ETC_PATH / "logs")


@router.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    """
    Embeds the user query, search Pinecone for relevant records, reranks the records and generate a response
    using OpenAI.

    Args:
        request (QueryRequest): The request object containing the query, client, and project information.

    Returns:
        QueryResponse: The response containing the generated answer and context from the matched records.
    """
    openai_client = get_openai_client()
    # Embed question
    embedded_queries = await embed_openai(openai_client, request.query, DIMENSIONS[ENV["EMBEDDING_MODEL"]])
    embedded_query: list[float] = embedded_queries[0]  # Only embedded 1 query

    # Query Pinecone
    index = await get_pinecone_index(request.client)
    query_results = index.query(
        namespace=request.project,
        vector=embedded_query,
        top_k=int(ENV["TOP_K"]),
        include_values=False,
        include_metadata=True,
    ).matches
    matches = [
        PineconeRecord(
            id=result.id,
            score=float(result.score),
            metadata=PineconeMetadata(**_process_metadata(result.metadata)),
        )
        for result in query_results
    ]
    logger.info(f"Retreived {len(matches)} matches from vector store.")
    ranked_records: list[PineconeRecord] = await rerank(get_cohere_client(), request.query, matches)

    # Pass prompt to LLM
    try:
        stream = openai_client.chat.completions.create(
            model=ENV["CHAT_MODEL"],
            messages=_construct_prompt(request, ranked_records),
            temperature=float(ENV["TEMPERATURE"]),
        )
        answer = stream.choices[0].message.content if stream.choices[0].message.content else "I don't know."  # Temp
    except UnauthorizedException:
        raise HTTPException(status_code=401, detail=f"Unauthorized key for Pinecone index for client {request.client}.")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Unauthorized key for OpenAI API.")

    logger.info(f"LLM answer:\n{stream}")
    logger.debug(f"Context used:\n {[str(record.metadata) for record in ranked_records]}")
    return QueryResponse(answer=answer, context=ranked_records)


def _process_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Process the metadata dictionary to ensure certain fields are of correctly typed.

    Args:
        metadata (Dict): The metadata dictionary.

    Returns:
        Dict: The processed metadata dictionary with correct field types.
    """
    # Ensure specific fields are integers
    metadata["page"] = int(metadata["page"])
    metadata["chunk_id"] = int(metadata["chunk_id"])
    metadata["timestamp"] = int(metadata["timestamp"])
    return metadata


def _construct_prompt(
    request: QueryRequest, records: list[PineconeRecord]
) -> list[ChatCompletionUserMessageParam | ChatCompletionSystemMessageParam]:
    """
    Constructs a prompt for the chat completion model based on the query request and ranked records.

    Args:
        request (QueryRequest): The request object containing the query and other relevant information.
        records (list[PineconeRecord]): A list of PineconeRecord instances to be used as context for the prompt.

    Returns:
        list[ChatCompletionUserMessageParam | ChatCompletionSystemMessageParam]: A list containing the system and user prompts for the chat completion model.
    """
    prompt_path = Path(__file__).parent.parent / "prompts"
    system_prompt_path = prompt_path / "system.json"
    user_prompt_path = prompt_path / "user.json"

    with open(system_prompt_path, "r") as file:
        system_content: str = json.load(file)["content"]
        system_prompt = ChatCompletionSystemMessageParam(role="system", content=system_content)
    with open(user_prompt_path, "r") as file:
        user_content: str = json.load(file)["content"].format(
            question=request.query,
            context=[str(record.metadata) for record in records],
            today=datetime.today().strftime("%Y-%m-%d"),
        )
        user_prompt = ChatCompletionUserMessageParam(role="user", content=user_content)
    logger.debug(f"Prompt created -\nSystem: {system_prompt}\nUser: {user_prompt}")
    return [system_prompt, user_prompt]
