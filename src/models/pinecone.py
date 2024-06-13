from dataclasses import dataclass


@dataclass
class PineconeMetadata:
    """
    Named dictionary used to store metadata for Pinecone entries
    """

    chunk_id: str
    name: str
    timestamp: int
    text: str
    page: int


@dataclass
class PineconeResults:
    id: str
    score: float
    metadata: PineconeMetadata
