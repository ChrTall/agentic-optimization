from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from agentic_optimization.configuration.llm_config import LLMConfig


def create_chat_model(config: LLMConfig) -> BaseChatModel:
    options = {
        "timeout": config.timeout_seconds,
        "max_retries": config.max_retries,
    }

    # Nur setzen, wenn ausdrücklich konfiguriert: Reasoning-Modelle
    # lehnen einen Temperaturparameter teilweise ab.
    if config.temperature is not None:
        options["temperature"] = config.temperature

    # Weniger „Nachdenken“ = deutlich schnellere Antworten bei
    # Reasoning-Modellen (z. B. gpt-5.x-mini, gpt-oss bei Groq).
    if config.reasoning_effort is not None and config.provider in {
        "openai", "groq",
    }:
        options["reasoning_effort"] = config.reasoning_effort

    return init_chat_model(
        model=config.model,
        model_provider=config.provider,
        api_key=config.api_key,
        **options,
    )
