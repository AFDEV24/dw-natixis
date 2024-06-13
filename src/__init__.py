from pathlib import Path

from src.utils.load_env import load_env_vars

etc_path = Path(__file__).parent.parent / "etc"

# Init env
ENV = load_env_vars(str(etc_path / ".env"))

DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "amazon.titan-embed-text-v1": 1536,
}
