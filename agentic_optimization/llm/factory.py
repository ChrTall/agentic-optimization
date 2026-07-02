from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from agentic_optimization.configuration.llm_config import LLMConfig


def create_chat_model(config: LLMConfig) -> BaseChatModel:
    return init_chat_model(
        model=config.model,
        model_provider=config.provider,
        api_key=config.api_key,
    )
