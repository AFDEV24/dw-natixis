import json

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam

from src import ENV
from src.utils.logger import get_logger
from src.utils.connections import OpenAIClient

ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
logger = get_logger(ETC_PATH / "logs")


@dataclass
class ChatExchange:
    question: str
    answer: str


class ChatHistory:
    def __init__(self, max_capacity: int = 8) -> None:
        self.recorded_histories: dict[str, deque[ChatExchange]] = {}
        self.max_capacity = max_capacity

    def get(self, chat_id: str) -> list[ChatExchange]:
        return list(self.recorded_histories.get(chat_id, []))

    def put(self, chat_id: str, chat_exchange: ChatExchange) -> None:
        exchanges = self.recorded_histories.get(chat_id, deque())
        exchanges.append(chat_exchange)

        if len(exchanges) > self.max_capacity:
            exchanges.popleft()
        self.recorded_histories[chat_id] = exchanges

    def exchanges_to_string(self, chat_id: str) -> str:
        exchanges = self.recorded_histories[chat_id]
        return "\n------\n".join(
            [f"User question: {exchange.question}\nAssistant answer: {exchange.answer}" for exchange in exchanges]
        )

    def reformulate_query(self, openai_client: OpenAIClient, chat_id: str, query: str) -> str:
        exchanges: str = self.exchanges_to_string(chat_id)
        answer = openai_client.chat.completions.create(
            model=ENV["REFORMULATION_MODEL"],
            messages=self._construct_prompt(exchanges, query),
            temperature=0.2,
        )
        return answer.choices[0].message.content if answer.choices[0].message.content else ""

    def _construct_prompt(
        self, exchanges: str, query: str
    ) -> list[ChatCompletionUserMessageParam | ChatCompletionSystemMessageParam]:
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
