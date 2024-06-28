from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from openai import AuthenticationError
from pinecone.exceptions import UnauthorizedException

from src import DIMENSIONS, ENV
from src.models.chat_history import ChatHistory, ChatExchange
from src.models.pinecone import PineconeRecord
from src.models.requests import QueryRequest
from src.models.response import QueryResponse
from src.utils.connections import get_cohere_client, get_openai_client
from src.utils.embeddings import embed_openai
from src.utils.logger import get_logger
from src.utils.llm_calls import rerank, openai_chat
from src.utils.prompt_builder import build_user_query_prompt, build_reformulation_prompt
from src.utils.pinecone import PineconeIndex

router = APIRouter()
ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
logger = get_logger(file_path=ETC_PATH / "logs")


chat_history = ChatHistory()


@router.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    """
    Reformulate user query with chat history, embeds the user query, search Pinecone for relevant records,
    reranks the records and generate a response using OpenAI.

    Args:
        request (QueryRequest): The request object containing the query, client, and project information.

    Returns:
        QueryResponse: The response containing the generated answer and context from the matched records.
    """
    openai_client = get_openai_client()

    user_query: str = request.query
    # Use chat history to reformulate question
    exchanges = chat_history.get(request.chat_id)
    if exchanges:
        formatted_exchanges = chat_history.format_exchanges(exchanges)
        user_query = await openai_chat(
            openai_client=openai_client,
            model=ENV["REFORMULATION_MODEL"],
            prompt=build_reformulation_prompt(exchanges=formatted_exchanges, query=request.query),
            temperature=float(ENV["TEMPERATURE"]),
            default_idk="",
        )
        logger.info(f"Reformulated user query from {request.query} -> {user_query}")
        logger.debug(f"Chat history of length {len(exchanges)} used:\n{formatted_exchanges}")
        if user_query == "":
            logger.error("Failed to reformulate user query!")
            raise HTTPException(status_code=500, detail="Reformulation failed.")

    # Embed question
    embedded_queries = await embed_openai(openai_client, user_query, DIMENSIONS[ENV["EMBEDDING_MODEL"]])
    embedded_query: list[float] = embedded_queries[0]  # Only embedded 1 query for now

    # Query Pinecone
    index = PineconeIndex(index=request.client)
    query_results = await index.query(request.project, embedded_query, top_k=int(ENV["TOP_K"]))
    records = index.to_records(query_results=query_results, cutoff=float(ENV["CUTOFF"]))
    if not records:
        return QueryResponse(answer="Could not retreive relavent context.", context=[])

    # Rerank
    ranked_records: list[PineconeRecord] = await rerank(get_cohere_client(), user_query, records)

    try:
        # Query LLM
        answer = await openai_chat(
            openai_client=openai_client,
            prompt=build_user_query_prompt(request, ranked_records),
            model=ENV["CHAT_MODEL"],
            temperature=float(ENV["TEMPERATURE"]),
        )
        if request.chat_id:
            chat_history.put(request.chat_id, chat_exchange=ChatExchange(question=request.query, answer=answer))
    except UnauthorizedException:
        raise HTTPException(status_code=401, detail=f"Unauthorized key for Pinecone index for client {request.client}.")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Unauthorized key for OpenAI API.")

    logger.info(f"LLM answer:\n\n{answer}")
    return QueryResponse(answer=answer, context=ranked_records)
