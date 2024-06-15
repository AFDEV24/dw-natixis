from dataclasses import dataclass, field, asdict

from src.models.base import FundMetadata


@dataclass
class PineconeMetadata(FundMetadata):
    chunk_id: int
    name: str
    timestamp: int
    text: str
    page: int

    def __str__(self) -> str:
        """
        Returns a formatted string representation of the PineconeMetadata fields.

        Returns:
            str: A formatted string containing all the fields of PineconeMetadata.
        """
        return "\n".join(f"{key}: {value}" for key, value in asdict(self).items())

    def to_string_except_text(self) -> str:
        """
        Returns a string representation of the PineconeMetadata fields, excluding the 'text' field.

        Returns:
            str: A formatted string containing all the fields of PineconeMetadata except 'text'.
        """
        metadata_dict = asdict(self)
        metadata_dict.pop("text")
        return "\n".join(f"{key}: {value}" for key, value in metadata_dict.items())


@dataclass
class PineconeRecord:
    id: str
    metadata: PineconeMetadata
    values: list[float] = field(default_factory=list)
    score: float = field(default_factory=float)
