import tiktoken
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Iterable

from src import ENV
from src.utils.logger import get_logger
from src.utils.tokens import count_tokens


ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
logger = get_logger(ETC_PATH / "logs")


@dataclass(frozen=True)
class ChatExchange:
    question: str
    answer: str


class ChatHistory:
    """
    Class to manage chat history, ensuring the total tokens of stored exchanges do not exceed a specified limit.

    Attributes:
        max_tokens (int): The maximum number of tokens allowed in the chat history.
        recorded_histories (dict): A dictionary storing chat histories with chat IDs as keys.
        tokenizer: The tokenizer used to count tokens in the chat exchanges.
    """

    def __init__(self, max_tokens: int = 60000) -> None:
        self.max_tokens = max_tokens
        self.recorded_histories: dict[str, deque[ChatExchange]] = {}

    def get(self, chat_id: str) -> list[ChatExchange]:
        """
        Retrieves the chat history for a given chat ID.
        """
        return list(self.recorded_histories.get(chat_id, []))

    def put(self, chat_id: str, chat_exchange: ChatExchange) -> None:
        """
        Adds a new chat exchange to the history for a given chat ID and ensures the token limit is not exceeded.
        """
        exchanges = self.recorded_histories.get(chat_id, deque())
        exchanges.append(chat_exchange)

        model_name = ENV["REFORMULATION_MODEL"]
        token_count: int = sum(
            [count_tokens(model_name, [exchange.question, exchange.answer]) for exchange in exchanges]
        )
        while token_count > self.max_tokens:
            oldest_exchange = exchanges.popleft()
            oldest_exchange_token_count = count_tokens(model_name, [oldest_exchange.question, oldest_exchange.answer])
            token_count -= oldest_exchange_token_count

        self.recorded_histories[chat_id] = exchanges

    def format_exchanges(self, exchanges: Iterable[ChatExchange]) -> str:
        """
        Formats a list of chat exchanges into a string with each exchange separated by a delimiter.
        """
        return "\n------\n".join(
            [f"User question: {exchange.question}\nAssistant answer: {exchange.answer}" for exchange in exchanges]
        )
