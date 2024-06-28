from pydantic import BaseModel

from src.models.base import FundMetadata


class CreateEmbeddingsRequest(BaseModel, FundMetadata):
    client: str
    project: str


class QueryRequest(BaseModel):
    client: str
    project: str
    query: str
    chat_id: str | None = None
