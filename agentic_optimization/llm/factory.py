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

    return init_chat_model(
        model=config.model,
        model_provider=config.provider,
        api_key=config.api_key,
        **options,
    )
