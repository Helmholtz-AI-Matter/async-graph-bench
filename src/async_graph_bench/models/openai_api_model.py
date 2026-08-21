import asyncio
import os
import json
from typing import Dict
from types import SimpleNamespace
from openai import AsyncOpenAI

from async_graph_bench.models.openai_api_response_wrapper import (
    OpenAIAPIResponseWrapper,
)

from async_graph_bench.models.abstract_classes import Model
from async_graph_bench.models.generation_parameters import GenerationParameters


def _should_dryrun() -> bool:
    """Check if dryrun mode is enabled via environment variable."""
    return os.environ.get("ASYNC_GRAPH_DRYRUN") == "1"


def _print_curl_command(
    url: str, headers: dict, payload: dict, enforce_url_suffix: bool = True
) -> None:
    """Print a curl command that replicates the API call."""
    # Build curl headers
    curl_headers = []
    for header, value in headers.items():
        if "authorization" in header.lower():
            curl_headers.append('-H "Authorization: Bearer ${OPENAI_API_KEY}" \\')
        else:
            curl_headers.append(f"-H '{header}: {value}' \\")

    if enforce_url_suffix:
        if not (url.endswith("completions") or url.endswith("completions/")):
            suffix = "chat/completions" if url.endswith("/") else "/chat/completions"
            url = url + suffix
    # Build curl command
    curl_parts = [
        f"curl '{url}' \\",
        "\n".join(curl_headers),
        #        f"-X POST \\",
        f"-d '{json.dumps(payload, indent=2)}'",
    ]

    curl_cmd = "\n".join(curl_parts)
    print(f"\n{'=' * 80}")
    print("DRYRUN - HTTP Request that would be made:")
    print(f"{'=' * 80}")
    print(f"URL: {url}")
    print(f"Payload:\n{json.dumps(payload, indent=2)}")
    print("\nCurl command:")
    print(curl_cmd)
    print(f"{'=' * 80}\n")


def sampling_params_from_generation_params(
    generation_params: GenerationParameters,
) -> Dict:
    params = generation_params.to_dict()
    if "logprobs" in params:
        if params["logprobs"] != 0:
            params["top_logprobs"] = min(params["logprobs"], 20)
        params["logprobs"] = bool(params["logprobs"])
    return params


def _create_mock_response(use_chat: bool, has_logprobs: bool, n_choices: int = 1):
    """Create mock OpenAI response for dryrun mode."""
    choices = []
    for _ in range(n_choices):
        choice = SimpleNamespace()
        if use_chat:
            choice.message = SimpleNamespace(
                content="mock response", reasoning_content=None
            )
        else:
            choice.text = "mock response"
        choice.finish_reason = "mock_complete"
        choice.token_ids = [0]
        if has_logprobs:
            choice.logprobs = SimpleNamespace(content=[])
        else:
            choice.logprobs = None
        choices.append(choice)

    response = SimpleNamespace(choices=choices)
    return response


class OpenAIAPIModel(Model):
    def __init__(
        self,
        model_id: str,
        openai_api_key: str = None,
        openai_endpoint: str = None,
        limit_logprobs: bool = False,
        use_chat: bool = True,
    ):
        """
        use_chat: True => call chat.completions.create with `messages=...`
                  False => call completions.create with `prompt=...`
        """
        self.model_id = model_id
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", None)
        base_url = openai_endpoint or os.environ.get("OPENAI_BASE_URL", None)
        assert api_key is not None, "No API key provided"
        self.openai_api = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=5)
        self.limit_logprobs = limit_logprobs
        self.use_chat = use_chat

        # pick the correct underlying create function and kw name once
        if self.use_chat:
            self._create = self.openai_api.chat.completions.create
            self._payload_kw = "messages"
        else:
            self._create = self.openai_api.completions.create
            self._payload_kw = "prompt"

    async def query(
        self, prompt, generation_params: GenerationParameters
    ) -> OpenAIAPIResponseWrapper:
        wrapper = OpenAIAPIResponseWrapper(
            n_logprobs=generation_params.logprobs
            if self.limit_logprobs and hasattr(generation_params, "logprobs")
            else None
        )

        params = sampling_params_from_generation_params(generation_params)

        def messages_to_text(messages_list):
            return "\n".join(m.get("content", "") for m in messages_list)

        # normalize into payloads appropriate for the chosen API:
        # - chat -> payloads is a list where each element is a messages list (List[Dict])
        # - completions -> payloads is a list where each element is a prompt string
        payloads = []

        if isinstance(prompt, str):
            payloads = (
                [{"role": "user", "content": prompt}] if self.use_chat else prompt
            )
            # wrap into list-of-payloads
            payloads = [payloads]

        elif isinstance(prompt, list):
            if not prompt:
                raise ValueError("Empty prompt list is not supported")

            # list[str]
            if all(isinstance(item, str) for item in prompt):
                if self.use_chat:
                    # IMPORTANT: each item must be a messages *list* (not a dict)
                    payloads = [[{"role": "user", "content": s}] for s in prompt]
                else:
                    payloads = list(prompt)

            # list[dict] -> treat as a single messages list (chat) or convert to one prompt (completions)
            elif all(isinstance(item, dict) for item in prompt):
                if self.use_chat:
                    payloads = [prompt]  # single messages list
                else:
                    payloads = [messages_to_text(prompt)]

            # list[list[dict]] -> multiple messages payloads
            elif all(
                isinstance(item, list)
                and item
                and all(isinstance(m, dict) for m in item)
                for item in prompt
            ):
                if self.use_chat:
                    payloads = list(prompt)  # already list of messages lists
                else:
                    payloads = [messages_to_text(item) for item in prompt]
            else:
                raise ValueError(
                    "Invalid prompt shape. Accepts: str, list[str], list[dict], or list[list[dict]]."
                )
        else:
            raise ValueError("Unsupported prompt type")

        # Build and run tasks
        base_params = {**params, "model": self.model_id}
        tasks = []
        for p in payloads:
            if self.use_chat:
                call_kwargs = {**base_params, "messages": p}
            else:
                call_kwargs = {**base_params, "prompt": p}

            # Dryrun mode: print curl command instead of making request
            if _should_dryrun():
                headers = {
                    "Authorization": f"Bearer {self.openai_api.api_key}",
                    "Content-Type": "application/json",
                }
                _print_curl_command(str(self.openai_api.base_url), headers, call_kwargs)
                # Skip actual HTTP call - return mock response which will cause the benchmark to fail
                mock_response = _create_mock_response(
                    use_chat=self.use_chat, has_logprobs=params.get("logprobs", 0) > 0
                )
                tasks.append(
                    asyncio.create_task(asyncio.sleep(0, result=mock_response))
                )
            else:
                tasks.append(self._create(**call_kwargs))

        responses = await asyncio.gather(*tasks)
        for r in responses:
            wrapper.append_response(r)
        return wrapper
