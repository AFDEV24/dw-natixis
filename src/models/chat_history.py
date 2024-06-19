import json

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam

from src import ENV
from src.utils.clients import OpenAIClient


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
        history = self.recorded_histories[chat_id]
        history.append(chat_exchange)

        if len(history) > self.max_capacity:
            history.popleft()
            self.recorded_histories[chat_id] = history

    def exchanges_to_string(self, chat_id: str) -> str:
        history = self.recorded_histories[chat_id]
        return "\n\n".join(
            [f"User question: {exchange.question}\nAssistant answer: {exchange.answer}" for exchange in history]
        )

    def reformulate_query(self, openai_client: OpenAIClient, chat_id: str, query: str) -> str:
        exchanges: str = self.exchanges_to_string(chat_id)
        answer = openai_client.chat.completions.create(
            model=ENV["REFORMULATION_MODEL"],
            messages=self._construct_prompt(exchanges, query),
            temperature=0,
        )
        return answer.choices[0].message.content if answer.choices[0].message.content else ""

    def _construct_prompt(
        self, exchanges: str, query: str
    ) -> list[ChatCompletionUserMessageParam | ChatCompletionSystemMessageParam]:
        """
        Constructs a prompt for the chat completion model based on the query request and ranked records.

        Args:
            request (QueryRequest): The request object containing the query and other relevant information.
            records (list[PineconeRecord]): A list of PineconeRecord instances to be used as context for the prompt.

        Returns:
            list[ChatCompletionUserMessageParam | ChatCompletionSystemMessageParam]: A list containing the system and user prompts for the chat completion model.
        """
        prompt_path = Path(__file__).parent.parent / "chat_history" / "prompts"
        system_prompt_path = prompt_path / "system.json"
        user_prompt_path = prompt_path / "user.json"

        with open(system_prompt_path, "r") as file:
            system_content: str = json.load(file)["content"]
            system_prompt = ChatCompletionSystemMessageParam(role="system", content=system_content)
        with open(user_prompt_path, "r") as file:
            user_content: str = json.load(file)["content"].format(
                question=query,
                history=exchanges,
            )
            user_prompt = ChatCompletionUserMessageParam(role="user", content=user_content)
        return [system_prompt, user_prompt]
