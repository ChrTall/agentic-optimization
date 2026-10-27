import os
from dataclasses import dataclass

from dotenv import load_dotenv

VALID_LLM_PROVIDERS = [
    "anthropic",
    "groq",
    "openai",
]


@dataclass(frozen=True)
class LLMConfig:
    """Configuration required to create an LLM client.

    Attributes:
        provider: Name of the selected LLM provider.
        model: Model name used for the selected provider.
        api_key: API key for authenticating with the selected provider.
    """
    provider: str
    model: str
    api_key: str
    timeout_seconds: float = 60.0
    max_retries: int = 2
    temperature: float | None = None
    reasoning_effort: str | None = None


VALID_REASONING_EFFORTS = ["none", "minimal", "low", "medium", "high"]


def load_llm_configuration(load_env_file: bool = True) -> LLMConfig:
    """Load and validate the LLM configuration from environment variables.

    Environment variables can either be provided directly by the system or
    through a local `.env` file. This makes the setup usable for both local
    development and containerized deployments.

    Required environment variables:
        - LLM_PROVIDER: Name of the LLM provider. Must be one of anthropic, groq, or openai.
        - LLM_MODEL: Model name for the selected provider.
        - LLM_API_KEY: API key for the selected provider.
    Args:
        load_env_file: Whether to load variables from a local `.env` file.
            This should usually be enabled for local development.

    Returns:
        Validated LLM configuration.

    Raises:
        ValueError: If one of the required environment variables is missing
            or invalid.
    """
    if load_env_file:
        load_dotenv()
    llm_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if llm_provider not in VALID_LLM_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER environment variable must be one of: {VALID_LLM_PROVIDERS}")
    llm_model = os.getenv("LLM_MODEL", "").strip()
    if not llm_model:
        raise ValueError(f"LLM_MODEL environment variable must contain a valid model for your LLM_PROVIDER: {llm_provider}")
    llm_api_key = os.getenv("LLM_API_KEY", "").strip()
    if not llm_api_key:
        raise ValueError(f"LLM_API_KEY environment variable must contain a valid API key for your LLM_PROVIDER: {llm_provider}.")
    timeout_seconds = _read_float("LLM_TIMEOUT_SECONDS", 60.0, minimum=1.0)
    max_retries = int(_read_float("LLM_MAX_RETRIES", 2, minimum=0))
    temperature_text = os.getenv("LLM_TEMPERATURE", "").strip()
    temperature = (
        _read_float("LLM_TEMPERATURE", 0.0, minimum=0.0)
        if temperature_text
        else None
    )
    reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "").strip().lower() or None
    if reasoning_effort is not None and reasoning_effort not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"LLM_REASONING_EFFORT must be one of: {VALID_REASONING_EFFORTS}"
        )
    print(f"Using LLM Provider: {llm_provider}, Model: {llm_model}")
    return LLMConfig(
        provider=llm_provider,
        model=llm_model,
        api_key=llm_api_key,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )


def _read_float(name: str, default: float, *, minimum: float) -> float:
    """Liest eine optionale Zahl aus der Umgebung."""

    raw_value = os.getenv(name, "").strip()

    if not raw_value:
        return float(default)

    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} muss eine Zahl sein.") from error

    if value < minimum:
        raise ValueError(f"{name} muss mindestens {minimum:g} sein.")

    return value
