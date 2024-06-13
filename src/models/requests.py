from pydantic import BaseModel


class CreateEmbeddingsRequest(BaseModel):
    client: str
    project: str
    # urls: list[str]


class QueryRequest(BaseModel):
    client: str
    project: str
    query: str
    file_name: str | None = None
