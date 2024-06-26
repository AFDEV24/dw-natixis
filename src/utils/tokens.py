import tiktoken


def count_tokens(model: str, prompts: list[str]) -> int:
    """
    Calculate the total number of tokens for a given model and list of prompts.

    Parameters:
    - model (str): The name of the model to use for tokenization.
    - prompts (list[str]): A list of strings representing prompts to tokenize.

    Returns:
    - int: The total number of tokens across all prompts after tokenization.
    """
    tokenizer = tiktoken.encoding_for_model(model)
    token_counts = [len(tokenizer.encode(prompt)) for prompt in prompts]
    return sum(token_counts)
