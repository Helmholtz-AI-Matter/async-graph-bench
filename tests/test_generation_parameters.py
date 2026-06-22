import pytest
from async_graph_bench import GenerationParameters


class TestGenerationParameters:
    def test_all_none_init(self):
        gp = GenerationParameters()
        assert gp.to_dict() == {}

    def test_single_param(self):
        gp = GenerationParameters(temperature=0.7)
        assert gp.to_dict() == {"temperature": 0.7}

    def test_multiple_params(self):
        gp = GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            n=3,
        )
        d = gp.to_dict()
        assert d["temperature"] == 0.7
        assert d["top_p"] == 0.9
        assert d["max_tokens"] == 100
        assert d["n"] == 3

    def test_none_values_excluded(self):
        gp = GenerationParameters(temperature=0.7, top_p=None, max_tokens=50)
        d = gp.to_dict()
        assert "top_p" not in d
        assert "temperature" in d
        assert "max_tokens" in d

    def test_to_dict_returns_copy(self):
        gp = GenerationParameters(temperature=0.7)
        d1 = gp.to_dict()
        d1["temperature"] = 0.0
        d2 = gp.to_dict()
        assert d2["temperature"] == 0.7

    def test_adapt_for_model_full_mapping(self):
        gp = GenerationParameters(temperature=0.7, top_p=0.9)
        mapping = {"temperature": "temp", "top_p": "topP"}
        result = gp.adapt_for_model(mapping)
        assert result == {"temp": 0.7, "topP": 0.9}

    def test_adapt_for_model_partial_mapping(self):
        gp = GenerationParameters(temperature=0.7, top_p=0.9)
        mapping = {"temperature": "temp"}
        result = gp.adapt_for_model(mapping)
        assert result == {"temp": 0.7}

    def test_adapt_for_model_filters_none_mapping(self):
        gp = GenerationParameters(temperature=0.7, top_p=0.9)
        mapping = {"temperature": "temp", "top_p": None}
        result = gp.adapt_for_model(mapping)
        assert result == {"temp": 0.7}

    def test_adapt_for_model_empty_mapping(self):
        gp = GenerationParameters(temperature=0.7)
        result = gp.adapt_for_model({})
        assert result == {}

    def test_stop_as_string(self):
        gp = GenerationParameters(stop="<END>")
        assert gp.to_dict()["stop"] == "<END>"

    def test_stop_as_list(self):
        gp = GenerationParameters(stop=["<END>", "<STOP>"])
        assert gp.to_dict()["stop"] == ["<END>", "<STOP>"]

    def test_seed(self):
        gp = GenerationParameters(seed=42)
        assert gp.to_dict()["seed"] == 42

    def test_logit_bias(self):
        gp = GenerationParameters(logit_bias={1: 0.5, 2: -1.0})
        assert gp.to_dict()["logit_bias"] == {1: 0.5, 2: -1.0}

    def test_allowed_token_ids(self):
        gp = GenerationParameters(allowed_token_ids=[100, 200, 300])
        assert gp.to_dict()["allowed_token_ids"] == [100, 200, 300]

    def test_response_format(self):
        gp = GenerationParameters(response_format={"type": "json_object"})
        assert gp.to_dict()["response_format"] == {"type": "json_object"}

    def test_all_params(self):
        gp = GenerationParameters(
            n=2,
            best_of=5,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            min_p=0.1,
            presence_penalty=0.2,
            frequency_penalty=0.1,
            repetition_penalty=1.1,
            seed=123,
            stop=["<END>"],
            stop_token_ids=[100],
            bad_words=["bad"],
            ignore_eos=True,
            max_tokens=200,
            min_tokens=10,
            logprobs=5,
            prompt_logprobs=3,
        )
        d = gp.to_dict()
        assert len(d) == 18
