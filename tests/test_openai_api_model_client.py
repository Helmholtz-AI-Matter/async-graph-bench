"""OpenAIAPIModel: max_retries and timeout of the underlying OpenAI client are configurable (no network)."""

import openai
import pytest

from async_graph_bench.models.openai_api_model import OpenAIAPIModel


def make(**kwargs):
    return OpenAIAPIModel(
        "some-model",
        openai_api_key="test-key",
        openai_endpoint="http://localhost:1",
        **kwargs,
    )


def test_defaults_unchanged():
    client = make().openai_api
    assert client.max_retries == 5
    assert client.timeout == openai.DEFAULT_TIMEOUT


def test_custom_max_retries_and_timeout():
    client = make(max_retries=12, timeout=42.5).openai_api
    assert client.max_retries == 12
    assert client.timeout == pytest.approx(42.5)


def test_zero_retries():
    assert make(max_retries=0).openai_api.max_retries == 0
