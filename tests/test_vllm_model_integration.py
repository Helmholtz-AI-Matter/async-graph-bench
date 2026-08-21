import pytest

vllm = pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

# Skip CPU-only tests as vLLM has platform detection issues on CPU
# and these tests require GPU acceleration to be practical
TESTDEVICE=torch.device("cuda:0")
if not torch.cuda.is_available():
    print("no GPU found, execution can be slow")
    TESTDEVICE=torch.device("cpu")

from vllm import LLM
from async_graph_bench.models.vllm_model import VLLMModel
from async_graph_bench import GenerationParameters


@pytest.fixture(scope="module")
def tiny_model_shared():
    """Load tiny model for integration tests that returns random outputs."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["HF_HOME"] = tmpdir
        os.environ["HF_HUB_CACHE"] = tmpdir

        #for tiny model alternatives check
        #https://github.com/vllm-project/vllm/blob/main/tests/models/registry.py
        #hinted at by
        #https://docs.vllm.ai/en/stable/contributing/model/tests/
        llm = LLM(model="yujiepan/mamba2-codestral-v0.1-tiny-random",
                  #max_seq_len=2048,
                  gpu_memory_utilization=0.5
                  )
        yield llm

        #try to cleanup
        del llm

@pytest.fixture
def tiny_model_instance():
    """Load tiny model for integration tests that returns random outputs."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["HF_HOME"] = tmpdir
        os.environ["HF_HUB_CACHE"] = tmpdir

        #for tiny model alternatives check
        #https://github.com/vllm-project/vllm/blob/main/tests/models/registry.py
        #hinted at by
        #https://docs.vllm.ai/en/stable/contributing/model/tests/
        llm = LLM(model="yujiepan/mamba2-codestral-v0.1-tiny-random",
                  #max_seq_len=2048,
                  gpu_memory_utilization=0.5
                  )
        yield llm

        #try to cleanup
        del llm

@pytest.fixture
def generation_params():
    return GenerationParameters(
        temperature=0.0,
        max_tokens=50,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )


class TestVLLMModelIntegration:
    """Integration tests for VLLMModel using real distilgpt2 model."""

    def test_init_without_chat_template(self, tiny_model_shared):
        """Test VLLMModel initialization without chat template."""
        model = VLLMModel(tiny_model_shared, use_chat_template=False)

        assert model.model is not None
        assert model.use_chat_template is False
        assert model.reasoning_parser_mode is None
        assert model.chat_template is None

    def test_init_with_chat_template(self, tiny_model_shared):
        """Test VLLMModel initialization with chat template."""
        model = VLLMModel(tiny_model_shared, use_chat_template=True)

        assert model.model is not None
        assert model.use_chat_template is True
        assert model.reasoning_parser_mode is None
        assert model.chat_template is None

    def test_init_with_custom_chat_template(self, tiny_model_shared):
        """Test VLLMModel initialization with custom chat template."""
        custom_template = "{{ 'Hello ' + message['content'] }}"
        model = VLLMModel(
            tiny_model_shared, use_chat_template=True, chat_template=custom_template
        )

        assert model.chat_template == custom_template

    def test_init_with_reasoning_parser(self, tiny_model_shared):
        """Test VLLMModel initialization with reasoning parser mode."""
        model = VLLMModel(
            tiny_model_shared,
            use_chat_template=False,
            reasoning_parser_mode="gpt-oss",
        )

        assert model.reasoning_parser_mode == "gpt-oss"

    @pytest.mark.slow
    async def test_query_without_chat_template(self, tiny_model_shared, generation_params):
        """Test querying VLLMModel without chat template."""
        model = VLLMModel(tiny_model_shared, use_chat_template=False)

        prompt = "The capital of France is"
        result = await model.query(prompt, generation_params)

        assert result is not None
        assert len(result.get_messages()) == 1
        assert len(result.get_messages()[0]) > 0

    @pytest.mark.slow
    async def test_query_with_chat_template(self, tiny_model_shared, generation_params):
        """Test querying VLLMModel with chat template."""
        model = VLLMModel(tiny_model_shared, use_chat_template=True)

        prompt = [{"role": "user", "content": "What is 2+2?"}]
        result = await model.query(prompt, generation_params)

        assert result is not None
        assert len(result.get_messages()) == 1
        assert len(result.get_messages()[0]) > 0

    @pytest.mark.slow
    async def test_query_batch_without_chat_template(self, tiny_model_shared, generation_params):
        """Test batch querying VLLMModel without chat template."""
        model = VLLMModel(tiny_model_shared, use_chat_template=False)

        prompts = ["The capital of France is", "The capital of Germany is"]
        result = await model.query(prompts, generation_params)

        assert result is not None
        assert len(result.get_messages()) == 2

    @pytest.mark.slow
    async def test_query_batch_with_chat_template(self, tiny_model_shared, generation_params):
        """Test batch querying VLLMModel with chat template."""
        model = VLLMModel(tiny_model_shared, use_chat_template=True)

        prompts = [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is 3+3?"}],
        ]
        result = await model.query(prompts, generation_params)

        assert result is not None
        assert len(result.get_messages()) == 2

    @pytest.mark.slow
    async def test_query_with_reasoning_parser(self, tiny_model_shared):
        """Test querying VLLMModel with reasoning parser and logprobs."""
        params = GenerationParameters(
            temperature=0.0,
            max_tokens=50,
            logprobs=1,
        )

        model = VLLMModel(
            tiny_model_shared,
            use_chat_template=False,
            reasoning_parser_mode="gpt-oss",
        )

        prompt = "The capital of France is"
        result = await model.query(prompt, params)

        assert result is not None
        assert len(result.get_messages()) == 1

    @pytest.mark.slow
    async def test_query_with_structured_output_json(self, tiny_model_shared, generation_params):
        """Test querying VLLMModel with JSON structured output."""
        params = GenerationParameters(
            temperature=0.0,
            max_tokens=100,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            },
        )

        model = VLLMModel(tiny_model_shared, use_chat_template=False)

        prompt = "Answer the following: What is 2+2?"
        result = await model.query(prompt, params)

        assert result is not None
        assert len(result.get_messages()) == 1

    @pytest.mark.slow
    async def test_query_with_structured_output_choice(self, tiny_model_shared, generation_params):
        """Test querying VLLMModel with choice structured output."""
        params = GenerationParameters(
            temperature=0.0,
            max_tokens=10,
            response_format={
                "type": "choice",
                "choice": ["Paris", "London", "Berlin", "Rome"],
            },
        )

        model = VLLMModel(tiny_model_shared, use_chat_template=False)

        prompt = "The capital of France is:"
        result = await model.query(prompt, params)

        assert result is not None
        assert len(result.get_messages()) == 1

    @pytest.mark.slow
    async def test_query_with_structured_output_regex(self, tiny_model_shared, generation_params):
        """Test querying VLLMModel with regex structured output."""
        params = GenerationParameters(
            temperature=0.0,
            max_tokens=20,
            response_format={
                "type": "regex",
                "regex": r"\d+",
            },
        )

        model = VLLMModel(tiny_model_shared, use_chat_template=False)

        prompt = "Compute: 2+2="
        result = await model.query(prompt, params)

        assert result is not None
        assert len(result.get_messages()) == 1

    @pytest.mark.slow
    def test_close_method(self, tiny_model_instance):
        """Test VLLMModel close method."""
        model = VLLMModel(tiny_model_instance, use_chat_template=False)

        model.close()

        assert model is not None

    async def test_disable_reasoning_parser(self, tiny_model_shared, generation_params):
        """Test disabling reasoning parser at query time."""
        model = VLLMModel(
            tiny_model_shared,
            use_chat_template=False,
            reasoning_parser_mode="gpt-oss",
        )

        params = GenerationParameters(
            temperature=0.0,
            max_tokens=50,
            logprobs=1,
        )

        prompt = "The capital of France is"
        result = await model.query(prompt, params, disable_reasoning_parser=True)

        assert result is not None
        assert result.reasoning_parser_mode is None
