import hashlib


def hash_string(s: str) -> str:
    """
    Generates a SHA-256 hash for the given string. The input string is encoded to UTF-8 bytes before hashing.

    Args:
        s (str): The input string to be hashed.

    Returns:
        str: The hexadecimal representation of the SHA-256 hash of the input string.

    Example:
        >>> hash_string("建築基準法施行令")
        '2f8f30b6b4fb29278d80255601dcf6d8a65a7dd5e8a6dfd3c1b3e8f9a206e1c5'
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(s.encode("utf-8"))
    return sha256_hash.hexdigest()
