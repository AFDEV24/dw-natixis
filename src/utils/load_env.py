from pathlib import Path

from dotenv import dotenv_values


def load_env_vars(env_file: str) -> dict[str, str]:
    """
    Load environment variables from a .env file and store them in a dictionary.

    Args:
        env_file (str): The path to the .env file.

    Returns:
        dict[str, str]: A dictionary containing the environment variables from the .env file.

    Raises:
        EnvironmentError: If any required environment variables are not set (i.e., have a value of None).
    """
    env = dotenv_values(dotenv_path=Path(env_file))
    validated_vars: dict[str, str] = {key: val for key, val in env.items() if val}
    if len(env.keys()) != len(validated_vars.keys()):
        raise EnvironmentError(
            f"Missing required environment variables: {[k for k in env.keys() if k not in validated_vars.keys()]}"
        )

    return validated_vars
