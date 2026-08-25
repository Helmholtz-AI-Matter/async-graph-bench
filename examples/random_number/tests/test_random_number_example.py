import pytest

from examples.random_number.run import (
    LengthExtractor,
    ResponseGenerator,
    extract_comma_separated,
    extract_from_numbered_list,
)
from examples.test_support import FakeModel


def test_extractors():
    assert extract_comma_separated("1, 2, 3") == 3
    assert extract_from_numbered_list("1. 10\n2. 20") == 2


@pytest.mark.asyncio
async def test_nodes_with_fake_model():
    model = FakeModel(["1, 2, 3"])
    response = await ResponseGenerator()(  # type: ignore[call-arg]
        {"prompt": ["return three numbers"]}, model
    )
    result = LengthExtractor()(
        {"response": response["response"], "extractor": [extract_comma_separated]}
    )

    assert result == {"length": [3]}
    assert model.prompts


@pytest.mark.slow
@pytest.mark.asyncio
async def test_nodes_with_tiny_random_llm(tiny_vllm_model):
    response = await ResponseGenerator()(  # type: ignore[call-arg]
        {"prompt": ["Return a short response"]}, tiny_vllm_model
    )

    assert response["response"]
