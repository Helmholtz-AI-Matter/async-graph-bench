import asyncio
from typing import List, Tuple
from async_graph_bench import GenerationParameters
from async_graph_bench.models.abstract_classes import Model, ResponseWrapper


QUESTIONS = {
    "What is the origin of Lorem Ipsum?": "Lorem Ipsum is simply dummy text of the printing and typesetting industry. It has been the industry's standard dummy text ever since the 1500s.",
    "Explain the water cycle in simple terms.": "The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow in clouds, and falls again to the surface as precipitation.",
    "What is the capital of France?": "The capital of France is Paris. It is located in the north-central part of the country along the Seine River.",
    "Describe how photosynthesis works.": "Photosynthesis is the process by which green plants convert light energy into chemical energy. Plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
    "What year did the Renaissance begin?": "The Renaissance began in the 14th century in Italy, marking a period of great cultural, artistic, and intellectual revival in Europe.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    "Sed ut perspiciatis unde omnis iste natus error sit voluptatem.": "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.",
    "Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet.": "Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem.",
    "Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis.": "Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur. Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse.": "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur, excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
}

DEFAULT_ANSWER = "Natus error sit voluptatem accusantium doloremque laudantium."


class MockResponseWrapper(ResponseWrapper):
    """Mock implementation of ResponseWrapper for testing."""

    def __init__(self, messages: List[str]):
        self._messages = messages
        self._tokens_per_message = [list(m) for m in messages]
        self._token_ids_per_message = [[ord(c) for c in m] for m in messages]
        self._logprobs_per_message = [[-0.1] * len(m) for m in messages]

    def get_messages(self) -> List[str]:
        return list(self._messages)

    def get_assistant_messages(self) -> List[str]:
        return list(self._messages)

    def get_reasoning_messages(self) -> List[List[str]]:
        return [[] for _ in self._messages]

    def get_tokens(self) -> List[List[str]]:
        return list(self._tokens_per_message)

    def get_reasoning_tokens(self) -> List[List[List[str]]]:
        return [[] for _ in self._messages]

    def get_assistant_tokens(self) -> List[List[str]]:
        return list(self._tokens_per_message)

    def get_token_ids(self) -> List[List[int]]:
        return list(self._token_ids_per_message)

    def get_reasoning_token_ids(self) -> List[List[List[int]]]:
        return [[] for _ in self._messages]

    def get_assistant_token_ids(self) -> List[List[int]]:
        return list(self._token_ids_per_message)

    def get_logprobs(self) -> List[List[float]]:
        return list(self._logprobs_per_message)

    def get_reasoning_logprobs(self) -> List[List[List[float]]]:
        return [[] for _ in self._messages]

    def get_assistant_logprobs(self) -> List[List[float]]:
        return list(self._logprobs_per_message)

    def get_tokens_alternatives(self) -> List[List[List[Tuple[str, float]]]]:
        return [[] for _ in self._messages]

    def get_reasoning_tokens_alternatives(self) -> List[List[List[List[Tuple[str, float]]]]]:
        return [[] for _ in self._messages]

    def get_assistant_tokens_alternatives(self) -> List[List[List[Tuple[str, float]]]]:
        return [[] for _ in self._messages]

    def get_finish_reasons(self) -> List[str]:
        return ["stop" for _ in self._messages]


class MockLLMModel(Model):
    """Mock LLM that responds with predefined answers."""

    def __init__(self, latency: float = 0.05):
        self._latency = latency

    async def query(self, prompt, generation_params: GenerationParameters) -> MockResponseWrapper:
        await asyncio.sleep(self._latency)

        messages = []
        if isinstance(prompt, str):
            messages.append(QUESTIONS.get(prompt, DEFAULT_ANSWER))
        elif isinstance(prompt, list):
            for item in prompt:
                if isinstance(item, str):
                    messages.append(QUESTIONS.get(item, DEFAULT_ANSWER))
                elif isinstance(item, dict):
                    content = item.get("content", "")
                    messages.append(QUESTIONS.get(content, DEFAULT_ANSWER))
                elif isinstance(item, list):
                    text_parts = [m.get("content", "") for m in item if isinstance(m, dict)]
                    text = "\n".join(text_parts)
                    messages.append(QUESTIONS.get(text, DEFAULT_ANSWER))
                else:
                    messages.append(DEFAULT_ANSWER)
        else:
            messages.append(DEFAULT_ANSWER)

        return MockResponseWrapper(messages)