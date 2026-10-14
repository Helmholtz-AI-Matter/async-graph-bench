import pytest

pytest.importorskip("openai")

from openai.types.chat import ChatCompletion

from async_graph_bench.models.openai_api_response_wrapper import (
    OpenAIAPIResponseWrapper,
    split_reasoning,
)


def make_completion(content, **message_fields):
    """A ChatCompletion as parsed by the openai client; unknown message fields end up in model_extra."""
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        **message_fields,
                    },
                }
            ],
        }
    )


def wrap(*completions):
    wrapper = OpenAIAPIResponseWrapper()
    for completion in completions:
        wrapper.append_response(completion)
    return wrapper


class TestSplitReasoning:
    def test_reasoning_content_field(self):
        message = make_completion("hi", reasoning_content="thought").choices[0].message
        assert split_reasoning(message) == ("thought", "hi")

    def test_reasoning_field(self):
        message = make_completion("\n\nhi", reasoning="thought").choices[0].message
        assert split_reasoning(message) == ("thought", "\n\nhi")

    def test_reasoning_content_takes_precedence(self):
        message = (
            make_completion("hi", reasoning_content="a", reasoning="b")
            .choices[0]
            .message
        )
        assert split_reasoning(message) == ("a", "hi")

    def test_inline_think_block(self):
        message = make_completion("<think>thought\n</think>\n\nhi").choices[0].message
        assert split_reasoning(message) == ("thought", "hi")

    def test_inline_think_without_opening_tag(self):
        message = make_completion("thought</think>hi").choices[0].message
        assert split_reasoning(message) == ("thought", "hi")

    def test_empty_reasoning_field_falls_back_to_inline(self):
        message = (
            make_completion("<think>thought</think>hi", reasoning="").choices[0].message
        )
        assert split_reasoning(message) == ("thought", "hi")

    def test_no_reasoning(self):
        message = make_completion("hi").choices[0].message
        assert split_reasoning(message) == ("", "hi")

    def test_none_content(self):
        message = make_completion(None, reasoning="thought").choices[0].message
        assert split_reasoning(message) == ("thought", "")


class TestResponseWrapperReasoning:
    def test_reasoning_and_messages_per_response(self):
        wrapper = wrap(
            make_completion("a1", reasoning_content="r1"),
            make_completion("a2", reasoning="r2"),
            make_completion("<think>r3</think>a3"),
            make_completion("a4"),
        )
        assert wrapper.get_reasoning_messages() == [["r1"], ["r2"], ["r3"], []]
        assert wrapper.get_messages() == ["a1", "a2", "a3", "a4"]
        assert wrapper.get_assistant_messages() == wrapper.get_messages()
