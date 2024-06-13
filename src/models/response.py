from pydantic import BaseModel
from src.models.pinecone import PineconeResults


class UploadResponse(BaseModel):
    signed_urls: list[str]
    details: str


class CreateEmbeddingsResponse(BaseModel):
    ids: dict[str, tuple[list[str], int]]
    details: str


class QueryResponse(BaseModel):
    answer: str
    context: list[PineconeResults]
