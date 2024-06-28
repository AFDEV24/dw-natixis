import time
from pathlib import Path
from typing import Any
from pinecone import ScoredVector, Vector
from functools import lru_cache


from src.models.pinecone import PineconeRecord, PineconeMetadata
from src.utils.logger import get_logger
from src.utils.connections import get_pinecone_client
from src.utils.decorators import async_retry
from src import ENV


ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
logger = get_logger(file_path=ETC_PATH / "logs")


@lru_cache(maxsize=12)
class PineconeIndex:
    def __init__(self, index: str, pool: int = 20) -> None:
        print("NEW INDEX CONN")
        self.client = get_pinecone_client()
        self.index = self.client.Index(name=index, pool=pool)

    @async_retry(logger=logger, max_attempts=2, initial_delay=1, backoff_base=2)
    async def query(
        self, namespace: str, embedded_query: list[float], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[ScoredVector]:
        start = time.time()
        query_results = self.index.query(
            namespace=namespace,
            vector=embedded_query,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            filters=filters,
        ).matches
        logger.info(f"Took {round(time.time()-start, 2)} to retreive {len(query_results)} matches from index.")
        return query_results

    @async_retry(logger=logger, max_attempts=2, initial_delay=1, backoff_base=2)
    async def upsert(self, vectors: list[Vector], namespace: str) -> None:
        """
        Upserts vectors to the Pinecone index.

        Args:
            vectors (List[Vector]): A list of vectors to be uploaded.
            namespace (str): The namespace/project in the Pinecone index where the vectors will be stored.

        Returns:
            None
        """
        start = time.time()
        self.index.upsert(vectors, namespace=namespace)
        logger.info(f"Took {round(time.time()-start, 2)}s to upsert {len(vectors)} entries to Index '{namespace}'.")

    @staticmethod
    def to_records(query_results: list[ScoredVector], cutoff: float | None = None) -> list[PineconeRecord]:
        """
        Converts query results into PineconeRecords. Optionally, apply a cutoff filter on similarity score.

        Args:
            query_results (list[ScoredVector]): Returns from a Pinecone index query.
            cutoff (float | None): Cutoff value for a result to be processed into a record.

        Returns:
            List of Pinecone records with correctly typed metadata
        """

        def _process_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
            """
            Process the metadata dictionary to ensure certain fields are of correctly typed.

            Args:
                metadata (Dict): The metadata dictionary.

            Returns:
                Dict: The processed metadata dictionary with correct field types.
            """
            # Ensure specific fields are integers
            metadata["page"] = int(metadata["page"])
            metadata["chunk_id"] = int(metadata["chunk_id"])
            metadata["timestamp"] = int(metadata["timestamp"])

            # Hack: rename date metadata field
            metadata["file_creation_date"] = metadata["date"]
            metadata.pop("date")
            return metadata

        records: list[PineconeRecord] = []
        for result in query_results:
            if cutoff is not None and result.score <= cutoff:
                continue  # Dont add record if it doesnt pass cutoff

            records.append(
                PineconeRecord(
                    id=result.id,
                    score=float(result.score),
                    metadata=PineconeMetadata(**_process_metadata(result.metadata)),
                )
            )
        logger.info(f"{len(records)} after applying cutoff of {ENV['CUTOFF']}.")
        return records
