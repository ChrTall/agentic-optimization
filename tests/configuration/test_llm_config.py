import pytest

from agentic_optimization.configuration.llm_config import (
    LLMConfig,
    load_llm_configuration,
)


def test_load_llm_configuration_returns_valid_config(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")

    config = load_llm_configuration(load_env_file=False)

    assert config == LLMConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="test-api-key",
    )


def test_load_llm_configuration_normalizes_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "OPENAI")
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")

    config = load_llm_configuration(load_env_file=False)

    assert config.provider == "openai"


def test_load_llm_configuration_strips_whitespace(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "  openai  ")
    monkeypatch.setenv("LLM_MODEL", "  gpt-4.1-mini  ")
    monkeypatch.setenv("LLM_API_KEY", "  test-api-key  ")

    config = load_llm_configuration(load_env_file=False)

    assert config == LLMConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="test-api-key",
    )


def test_load_llm_configuration_raises_for_missing_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        load_llm_configuration(load_env_file=False)


def test_load_llm_configuration_raises_for_invalid_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "invalid-provider")
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        load_llm_configuration(load_env_file=False)


def test_load_llm_configuration_raises_for_missing_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")

    with pytest.raises(ValueError, match="LLM_MODEL"):
        load_llm_configuration(load_env_file=False)


def test_load_llm_configuration_raises_for_empty_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")

    with pytest.raises(ValueError, match="LLM_MODEL"):
        load_llm_configuration(load_env_file=False)


def test_load_llm_configuration_raises_for_missing_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        load_llm_configuration(load_env_file=False)


def test_load_llm_configuration_raises_for_empty_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("LLM_API_KEY", "")

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        load_llm_configuration(load_env_file=False)
