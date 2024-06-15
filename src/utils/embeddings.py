from openai import Client as OpenAIClient

from src import ENV


async def embed_openai(client: OpenAIClient, text: str | list[str], dimensions: int) -> list[list[float]]:
    """
    Returns a list of list of vectors. The outter list is of the same size as input text list.
    """
    embeddings = client.embeddings.create(model=ENV["EMBEDDING_MODEL"], input=text, dimensions=dimensions).data
    return [emb.embedding for emb in embeddings]
