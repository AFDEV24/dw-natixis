from dataclasses import dataclass

from src.models.base import FundMetadata


@dataclass
class PineconeMetadata(FundMetadata):
    chunk_id: int
    name: str
    timestamp: int
    text: str
    page: int


@dataclass
class PineconeResults:
    id: str
    score: float
    metadata: PineconeMetadata
