import os
import subprocess
import sys

import pytest

from examples.multi_resource_test.query_model import QueryModel
from examples.test_support import FakeModel


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--models" in result.stdout


@pytest.mark.asyncio
async def test_query_model_with_fake_model():
    result = await QueryModel(max_tokens=4)(
        {"input_texts": ["What is two plus two?"]}, FakeModel(["4"])
    )

    assert result == {"greedy_texts": ["4"]}


@pytest.mark.slow
@pytest.mark.asyncio
async def test_query_model_with_tiny_random_llm(tiny_vllm_model):
    result = await QueryModel(max_tokens=4)(
        {"input_texts": ["Give a short answer."]}, tiny_vllm_model
    )

    assert result["greedy_texts"]
