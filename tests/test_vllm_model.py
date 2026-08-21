import pytest

pytest.importorskip("vllm")
pytest.importorskip("torch")

from async_graph_bench import GenerationParameters
from async_graph_bench.models import vllm_model
from async_graph_bench.models.vllm_model import (
    VLLMModel,
    sampling_params_from_generation_params,
)


class FakeLLM:
    def __init__(self):
        self.generate_calls = []
        self.chat_calls = []

    def generate(self, prompt, **kwargs):
        self.generate_calls.append((prompt, kwargs))
        return ["response"]

    def chat(self, prompt, **kwargs):
        self.chat_calls.append((prompt, kwargs))
        return ["response"]


class FakeResponseWrapper:
    def __init__(self, response, **kwargs):
        self.response = response


@pytest.fixture
def generation_params():
    return GenerationParameters(max_tokens=1)


async def test_generate_does_not_receive_chat_template(monkeypatch, generation_params):
    model = FakeLLM()
    monkeypatch.setattr(vllm_model, "VLLMResponseWrapper", FakeResponseWrapper)

    await VLLMModel(model, use_chat_template=False).query("prompt", generation_params)

    assert len(model.generate_calls) == 1
    assert "chat_template" not in model.generate_calls[0][1]


async def test_chat_receives_chat_template(monkeypatch, generation_params):
    model = FakeLLM()
    monkeypatch.setattr(vllm_model, "VLLMResponseWrapper", FakeResponseWrapper)

    await VLLMModel(
        model, use_chat_template=True, chat_template="custom-template"
    ).query("prompt", generation_params)

    assert len(model.chat_calls) == 1
    assert model.chat_calls[0][1]["chat_template"] == "custom-template"


async def test_chat_template_fallback_is_only_used_for_chat(
    monkeypatch, generation_params
):
    model = FakeLLM()
    calls = 0

    def chat_with_missing_template(prompt, **kwargs):
        nonlocal calls
        calls += 1
        model.chat_calls.append((prompt, kwargs))
        if calls == 1:
            raise RuntimeError("chat template is missing")
        return ["response"]

    model.chat = chat_with_missing_template
    monkeypatch.setattr(vllm_model, "VLLMResponseWrapper", FakeResponseWrapper)

    await VLLMModel(model, use_chat_template=True).query("prompt", generation_params)

    assert model.chat_calls[0][1]["chat_template"] is None
    assert model.chat_calls[1][1]["chat_template"] == "chatml.jinja"


def test_structured_output_uses_installed_vllm_parameter_name():
    params = GenerationParameters(
        response_format={
            "type": "json_schema",
            "json_schema": {"type": "object"},
        }
    )

    sampling_params = sampling_params_from_generation_params(params)

    structured_outputs = getattr(sampling_params, "structured_outputs", None)
    if structured_outputs is None:
        structured_outputs = getattr(sampling_params, "guided_decoding", None)

    assert structured_outputs is not None
