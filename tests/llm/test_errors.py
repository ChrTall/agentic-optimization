import httpx
import openai
import pytest

from agentic_optimization.llm.errors import (
    classify_llm_error,
    describe_llm_error,
    is_llm_service_error,
)

_REQUEST = httpx.Request("POST", "https://api.example.test/v1/chat")


def _status_error(error_class, status_code):
    response = httpx.Response(status_code, request=_REQUEST)
    return error_class("simulated", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (_status_error(openai.RateLimitError, 429), "rate_limit"),
        (_status_error(openai.AuthenticationError, 401), "authentication"),
        (_status_error(openai.NotFoundError, 404), "not_found"),
        (openai.APITimeoutError(request=_REQUEST), "timeout"),
        (openai.APIConnectionError(request=_REQUEST), "connection"),
    ],
)
def test_classifies_provider_errors(error, kind) -> None:
    assert is_llm_service_error(error)
    assert classify_llm_error(error) == kind
    assert describe_llm_error(error)


def test_ignores_ordinary_errors() -> None:
    assert not is_llm_service_error(ValueError("x"))
    assert classify_llm_error(RuntimeError("x")) == "other"
