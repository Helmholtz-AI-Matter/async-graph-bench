from async_graph_bench.models.reasoning_parsers import (
    parse_gpt_oss_reasoning,
    parse_reasoning,
    parse_deepseek_reasoning,
    find_subarray,
    find_subarray_backwards,
)


class TestFindSubarray:
    def test_found_at_start(self):
        assert find_subarray([1, 2, 3, 4], [1, 2]) == 0

    def test_found_in_middle(self):
        assert find_subarray([1, 2, 3, 4], [3, 4]) == 2

    def test_not_found(self):
        assert find_subarray([1, 2, 3], [4, 5]) == -1

    def test_empty_subarray(self):
        assert find_subarray([1, 2, 3], []) == 0

    def test_start_parameter(self):
        assert find_subarray([1, 2, 1, 2, 3], [1, 2], start=2) == 2

    def test_single_element(self):
        assert find_subarray([1, 2, 3], [2]) == 1


class TestFindSubarrayBackwards:
    def test_last_occurrence(self):
        tokens = [1, 2, 3, 1, 2, 3]
        assert find_subarray_backwards(tokens, [1, 2]) == 3

    def test_single_occurrence(self):
        assert find_subarray_backwards([1, 2, 3], [2, 3]) == 1

    def test_not_found(self):
        assert find_subarray_backwards([1, 2, 3], [4, 5]) == -1

    def test_range(self):
        tokens = [1, 2, 3, 1, 2, 3]
        assert find_subarray_backwards(tokens, [1, 2], start=0, end=3) == 0

    def test_empty_range(self):
        assert find_subarray_backwards([1, 2, 3], [1, 2, 3, 4]) == -1


class TestParseGptOssReasoning:
    def test_empty_tokens(self):
        result = parse_gpt_oss_reasoning([])
        assert result["reasoning"] == []
        assert result["message"] == (0, 0)

    def test_only_message(self):
        tokens = ["<|message|>", "hello", "world", "<|end|>"]
        result = parse_gpt_oss_reasoning(tokens)
        assert result["reasoning"] == []
        assert result["message"] == (1, 3)

    def test_only_reasoning_channel(self):
        tokens = ["<|channel|>", "thinking", "<|message|>", "think", "hard", "<|end|>"]
        result = parse_gpt_oss_reasoning(tokens)
        assert len(result["reasoning"]) == 1
        assert result["reasoning"][0][2] == "thinking"

    def test_reasoning_and_message(self):
        tokens = [
            "<|channel|>",
            "thinking",
            "<|message|>",
            "thinking text",
            "<|end|>",
            "<|channel|>",
            "final",
            "<|message|>",
            "answer text",
            "<|end|>",
        ]
        result = parse_gpt_oss_reasoning(tokens)
        assert len(result["reasoning"]) == 1
        assert result["message"] is not None

    def test_leading_orphan_tokens(self):
        tokens = [
            "orphan",
            "data",
            "<|channel|>",
            "analysis",
            "<|message|>",
            "t",
            "<|end|>",
        ]
        result = parse_gpt_oss_reasoning(tokens)
        assert len(result["reasoning"]) >= 1
        assert result["reasoning"][0][2] == "analysis"

    def test_multiple_reasoning_channels(self):
        tokens = [
            "<|channel|>",
            "analysis",
            "<|message|>",
            "a1",
            "<|end|>",
            "<|channel|>",
            "planning",
            "<|message|>",
            "p1",
            "<|end|>",
        ]
        result = parse_gpt_oss_reasoning(tokens)
        assert len(result["reasoning"]) == 2


class TestParseReasoning:
    def test_empty(self):
        result = parse_reasoning([], ["<start>"], ["<end>"])
        assert result["reasoning"] == [(0, 0, "thinking")]
        assert result["message"] == (0, 0)

    def test_only_reasoning(self):
        tokens = ["<start>", "think", "<end>"]
        result = parse_reasoning(tokens, ["<start>"], ["<end>"])
        assert len(result["reasoning"]) >= 1

    def test_reasoning_and_message(self):
        tokens = ["<start>", "thinking...", "<end>", "hello", "world"]
        result = parse_reasoning(tokens, ["<start>"], ["<end>"])
        assert len(result["reasoning"]) == 1
        assert result["message"][0] < len(tokens)

    def test_multiple_reasoning_segments(self):
        tokens = ["<start>", "r1", "<end>", "m1", "<start>", "r2", "<end>", "m2"]
        result = parse_reasoning(tokens, ["<start>"], ["<end>"])
        assert len(result["reasoning"]) == 2

    def test_no_tags_all_reasoning(self):
        tokens = ["just", "plain", "text"]
        result = parse_reasoning(tokens, ["<s>"], ["<e>"])
        assert len(result["reasoning"]) == 1
        assert result["reasoning"][0] == (0, 3, "thinking")


class TestParseDeepseekReasoning:
    def test_basic(self):
        tokens = ["\u200b", "thinking content", "</think>"]
        result = parse_deepseek_reasoning(tokens)
        assert isinstance(result, dict)
        assert "reasoning" in result
        assert "message" in result

    def test_no_thinking_tags(self):
        tokens = ["just", "normal", "response"]
        result = parse_deepseek_reasoning(tokens)
        assert len(result["reasoning"]) >= 1
