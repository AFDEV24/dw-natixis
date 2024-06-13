from functools import lru_cache
from pathlib import Path

from openai import Client as OpenAIClient
from langchain_community.embeddings import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore as LCPinecone
from pinecone import Index  # type: ignore
from pydantic.v1.types import SecretStr  # Langchain requires pydantic v1...
from typing import Any

from src.utils.logger import get_logger
from src import ENV


ETC_PATH: Path = Path(__file__).parent.parent.parent / "etc"
logger = get_logger(ETC_PATH / "logs")


@lru_cache(maxsize=8)
def get_lc_pinecone(index: Index, project: str, bedrock_client: Any | None) -> LCPinecone:  # tmp typing
    embeddings: BedrockEmbeddings | OpenAIEmbeddings
    if bedrock_client is not None:
        embeddings = BedrockEmbeddings(model_id=ENV["EMBEDDING_MODEL"], client=bedrock_client)
    else:
        embeddings = OpenAIEmbeddings(api_key=SecretStr(ENV["OPENAI_API_KEY"]), model=ENV["EMBEDDING_MODEL"])
    return LCPinecone(index=index, namespace=project, embedding=embeddings)


async def embed_openai(client: OpenAIClient, text: str | list[str], dimensions: int) -> list[list[float]]:
    """
    Returns a list of list of vectors. The outter list is of the same size as input text list.
    """
    embeddings = client.embeddings.create(model=ENV["EMBEDDING_MODEL"], input=text, dimensions=dimensions).data
    return [emb.embedding for emb in embeddings]


# from openai.types.embedding import Embedding
# from langchain.prompts import PromptTemplate
# from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
# from langchain_cohere.rerank import CohereRerank
# from langchain_core.documents import Document
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSerializable

# @lru_cache(maxsize=124)
# def create_rag_chain(
#     index: Index,
#     project: str,
#     file_name: str | None,
#     prompt_file: Path,
#     prompt_inputs: tuple[str],  # Needs to be hashable for caching
#     search_type: str = "similarity",
# ) -> RunnableSerializable:
#     """
#     Creates a retrieval-augmented generation (RAG) chain.

#     Args:
#         index (Index): The Pinecone index to use for retrieval.
#         project (str): The name of the project.
#         file_name (str): The name of the file to filter the search.
#         prompt_file (Path): The path to the prompt file.
#         prompt_inputs (tuple[str, ...]): The input variables for the prompt, must be hashable for caching.
#         search_type (str): The type of search to perform. Default is "similarity".

#     Returns:
#         RunnableSerializable: The constructed RAG chain.

#     This function performs the following steps:
#     1. Retrieves the retriever using the specified index and project.
#     2. Reads the prompt template from the specified file.
#     3. Creates a language model object used in the chain.
#     4. Constructs the RAG chain using the retriever, prompt template, and language model object.
#     5. Returns the constructed RAG chain.
#     """
#     top_k = int(ENV["TOP_K"])
#     search_criteria: dict[str, int | str | dict[str, str]] = {"k": top_k}
#     if file_name:
#         search_criteria["filter"] = {"name": file_name}

#     llm: ChatOpenAI | BedrockChat
#     if "gpt" in ENV["CHAT_MODEL"].lower():
#         aws_client = None
#         llm = ChatOpenAI(
#             api_key=SecretStr(ENV["OPENAI_API_KEY"]),
#             model=ENV["CHAT_MODEL"],
#             temperature=float(ENV["TEMPERATURE"]),
#         )
#     else:
#         aws_client = get_bedrock_client()
#         llm = BedrockChat(model_id=ENV["CHAT_MODEL"], client=aws_client)

#     retriever = get_lc_pinecone(index, project, aws_client).as_retriever(
#         search_type=search_type,
#         search_kwargs=search_criteria,
#     )
#     compressor = CohereRerank()

#     logger.info(f"Fetched top {top_k} chunks for file '{file_name}'")

#     with open(Path(__file__).parent.parent / "prompts" / prompt_file) as file:
#         prompt = file.read()
#         prompt_template = PromptTemplate(input_variables=list(prompt_inputs), template=prompt)

#     rag_chain_from_docs = (
#         RunnablePassthrough.assign(context=(lambda x: _combine_context(x["context"])))
#         | prompt_template
#         | llm
#         | StrOutputParser()
#     )

#     return RunnableParallel({"context": retriever, "question": RunnablePassthrough()}).assign(
#         answer=rag_chain_from_docs
#     )  # type: ignore


# def _combine_context(docs: list[Document]) -> str:
#     """
#     Combines the content of a list of documents into a single string for LLM injestion.
#     """
#     # TODO: Check token size of single string in case it's too big for LLM context window.
#     return "\n\n".join(doc.page_content for doc in docs)
