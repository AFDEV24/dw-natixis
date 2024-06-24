import json
from pathlib import Path
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from datetime import datetime

from src.models.pinecone import PineconeRecord
from src.models.requests import QueryRequest
from src.utils.logger import get_logger


ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
logger = get_logger(file_path=ETC_PATH / "logs")


def build_user_query_prompt(
    request: QueryRequest, records: list[PineconeRecord]
) -> list[ChatCompletionUserMessageParam | ChatCompletionSystemMessageParam]:
    """
    Constructs a prompt for the chat completion model based on the user query and retreived context.

    Args:
        request (QueryRequest): The request object containing the query and other relevant information.
        records (list[PineconeRecord]): A list of PineconeRecord instances to be used as context for the prompt.

    Returns:
        list[ChatCompletionUserMessageParam | ChatCompletionSystemMessageParam]:
        A list containing the system and user prompts for the chat completion model.
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "query"
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


def build_reformulation_prompt(
    exchanges: str, query: str
) -> list[ChatCompletionUserMessageParam | ChatCompletionSystemMessageParam]:
    """
    Constructs a prompt for reformulating a user query given the previous exchanges in chat history.

    Args:
        exchanges (str): Formatted history of exchanges between user and app.
        query (str): User posed query.

    Returns:
        list[ChatCompletionUserMessageParam | ChatCompletionSystemMessageParam]:
        A list containing the system and user prompts for the chat completion model.
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "chat_history"
    system_prompt_path = prompt_path / "system.json"
    user_prompt_path = prompt_path / "user.json"

    with open(system_prompt_path, "r") as file:
        system_content: str = json.load(file)["content"]
        system_prompt = ChatCompletionSystemMessageParam(role="system", content=system_content)
    with open(user_prompt_path, "r") as file:
        user_content: str = json.load(file)["content"].format(
            question=query,
            exchanges=exchanges,
        )
        user_prompt = ChatCompletionUserMessageParam(role="user", content=user_content)
    logger.debug(f"Prompt created for question reformuation -\nSystem: {system_prompt}\nUser: {user_prompt}")
    return [system_prompt, user_prompt]
