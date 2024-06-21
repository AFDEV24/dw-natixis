import hashlib


def hash_string(s: str) -> str:
    """
    Generates a SHA-256 hash for the given string. The input string is encoded to UTF-8 bytes before hashing.

    Args:
        s (str): The input string to be hashed.

    Returns:
        str: The hexadecimal representation of the SHA-256 hash of the input string.

    Example:
        >>> hash_string("helloworld")
        '936a185caaa266bb9cbe981e9e05cb78cd732b0b3280eb944412bb6f8f8f07af'
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(s.encode("utf-8"))
    return sha256_hash.hexdigest()
