import pytest


class FakeResponse:
    def __init__(self, messages, tokens=None):
        self._messages = messages
        self._tokens = tokens or [["mock"] for _ in messages]

    def get_messages(self):
        return self._messages

    def get_tokens(self):
        return self._tokens


class FakeModel:
    def __init__(self, messages=None):
        self.messages = messages or ["mock response"]
        self.prompts = []

    async def query(self, prompt, generation_params):
        self.prompts.append(prompt)
        messages = (
            self.messages[: len(prompt)] if isinstance(prompt, list) else self.messages
        )
        return FakeResponse(messages)


@pytest.fixture(scope="module")
def tiny_vllm_model():
    vllm = pytest.importorskip("vllm")
    from async_graph_bench.models.vllm_model import VLLMModel

    llm = vllm.LLM(
        model="yujiepan/mamba2-codestral-v0.1-tiny-random",
        max_model_len=256,
        max_num_seqs=1,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
    )
    model = VLLMModel(llm, use_chat_template=True)
    yield model
    model.close()
