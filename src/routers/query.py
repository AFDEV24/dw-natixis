import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from openai import AuthenticationError
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from pinecone.exceptions import UnauthorizedException

from src.models.requests import QueryRequest
from src.models.response import QueryResponse
from src.models.pinecone import PineconeResults, PineconeMetadata

from src.utils.embeddings import embed_openai
from src.utils.clients import get_pinecone_index, get_openai_client, get_cohere_client
from src.utils.logger import get_logger
from src.utils.parsers import rerank
from src import ENV, DIMENSIONS

router = APIRouter()
logger = get_logger(file_path=Path(__file__).parent.parent.parent / "etc" / "logs")


@router.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    """
    Handles a query request, invokes a retrieval-augmented generation (RAG) chain to provide an answer.

    Args:
        request (QueryRequest): The query request containing the client, project, file name, and query string.

    Returns:
        QueryResponse: The response containing the answer and the context used to generate the answer.

    This function performs the following steps:
    1. Retrieves the Pinecone index which serves as a connection using the provided client information.
    2. Creates a RAG chain using the retrieved index, project name, file name, and a prompt file.
    3. Invokes the chain with the query provided in the request.
    4. Extracts the answer and context from the response.
    5. Returns a QueryResponse object with the answer and context.
    """
    openai_client = get_openai_client()
    # Embed question
    embedded_queries = await embed_openai(openai_client, request.query, DIMENSIONS[ENV["EMBEDDING_MODEL"]])
    embedded_query: list[float] = embedded_queries[0]  # Only embedded 1 query

    # Query Pinecone
    index = await get_pinecone_index(client=request.client)
    query_results = index.query(
        namespace=request.project,
        vector=embedded_query,
        top_k=int(ENV["TOP_K"]),
        include_values=False,
        include_metadata=True,
    ).matches
    matches = [
        PineconeResults(
            id=result.id,
            score=float(result.score),
            metadata=PineconeMetadata(
                chunk_id=result.metadata["chunk_id"],
                name=result.metadata["name"],
                fund_name=result.metadata["fund_name"],
                timestamp=int(result.metadata["timestamp"]),
                text=result.metadata["text"],
                page=int(result.metadata["page"]),
                region=result.metadata["region"],
                date=result.metadata["date"],
            ),
        )
        for result in query_results
    ]
    logger.info(f"Retreived {len(matches)} matches from vector store.")
    # Reranking
    ranked_results: list[PineconeResults] = await rerank(get_cohere_client(), request.query, matches)
    relavent_context: list[dict[str, str | int]] = [
        {
            "text": res.metadata.text,
            "page_nunber": res.metadata.page,
            "file_name": res.metadata.name,
        }
        for res in ranked_results
    ]

    # Construct prompt
    system_prompt_path = Path(__file__).parent.parent / "prompts" / "system.json"
    user_prompt_path = Path(__file__).parent.parent / "prompts" / "user.json"

    with open(system_prompt_path, "r") as file:
        system_content: str = json.load(file)["content"]
        system_prompt = ChatCompletionSystemMessageParam(role="system", content=system_content)
    with open(user_prompt_path, "r") as file:
        user_content: str = json.load(file)["content"].format(question=request.query, context=relavent_context)
        user_prompt = ChatCompletionUserMessageParam(role="user", content=user_content)
    logger.debug(f"Prompt created -\nSystem: {system_prompt}\nUser: {user_prompt}")

    # Pass prompt to LLM
    try:
        stream = openai_client.chat.completions.create(
            model=ENV["CHAT_MODEL"], messages=[system_prompt, user_prompt], temperature=float(ENV["TEMPERATURE"])
        )  # type:ignore
        answer = stream.choices[0].message.content if stream.choices[0].message.content else "I don't know."
    except UnauthorizedException:
        raise HTTPException(status_code=401, detail=f"Unauthorized key for Pinecone index for client {request.client}.")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail=f"Unauthorized key for OpenAI API.")

    logger.debug(f"LLM answer:\n{stream}")
    logger.debug(f"Relavent context:\n{relavent_context}")
    return QueryResponse(answer=answer, context=ranked_results)
