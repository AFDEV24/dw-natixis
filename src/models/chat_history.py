import tiktoken
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Iterable

from src import ENV
from src.utils.logger import get_logger


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
        chat_length_limit_check (int): The maximum number of exchanges to check for token limits.
        recorded_histories (dict): A dictionary storing chat histories with chat IDs as keys.
        tokenizer: The tokenizer used to count tokens in the chat exchanges.
    """

    def __init__(self, max_tokens: int = 12000, chat_length_limit_check: int = 10) -> None:
        self.max_tokens = max_tokens
        self.chat_length_limit_check = chat_length_limit_check
        self.recorded_histories: dict[str, deque[ChatExchange]] = {}
        self.tokenizer = tiktoken.encoding_for_model(ENV["REFORMULATION_MODEL"])

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

        if len(exchanges) > self.chat_length_limit_check:
            while self._count_tokens(exchanges) > self.max_tokens:
                exchanges.popleft()
        self.recorded_histories[chat_id] = exchanges

    def _count_tokens(self, exchanges: deque[ChatExchange]) -> int:
        """
        Counts the total number of tokens in a list of chat exchanges.
        """
        total_tokens = 0
        for exchange in exchanges:
            total_tokens += len(self.tokenizer.encode(exchange.question)) + len(self.tokenizer.encode(exchange.answer))
        return total_tokens

    def format_exchanges(self, exchanges: Iterable[ChatExchange]) -> str:
        """
        Formats a list of chat exchanges into a string with each exchange separated by a delimiter.
        """
        return "\n------\n".join(
            [f"User question: {exchange.question}\nAssistant answer: {exchange.answer}" for exchange in exchanges]
        )
