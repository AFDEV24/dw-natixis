import asyncio
import logging
from functools import wraps

from fastapi import HTTPException


def async_retry(logger: logging.Logger, max_attempts: int = 3, initial_delay: int = 1, backoff_base: int = 2):
    """
    Decorator to retry an asynchronous task with exponential backoff.

    Args:
        max_attempts (int): Maximum number of retry attempts.
        initial_delay (int): Initial delay between retries in seconds.
        backoff_base (int): Factor by which to multiply the delay for each subsequent retry.

    Returns:
        Decorated function with retry capability.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            attempts = 0
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise HTTPException(status_code=500, detail=str(e)) from e
                    logger.warn(
                        f"{func.__name__}: Attempt {attempts} failed with error {e}. Retrying in {delay} seconds..."
                    )

                    await asyncio.sleep(delay)
                    delay += backoff_base**attempts

        return wrapper

    return decorator
