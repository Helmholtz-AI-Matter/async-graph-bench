from __future__ import annotations

import importlib.util

if importlib.util.find_spec("vllm") is None:
    raise ImportError(
        "To use this functionality, you need to install the 'vllm' module"
    )
if importlib.util.find_spec("torch") is None:
    raise ImportError(
        "To use this functionality, you need to install the 'torch' module"
    )

from models.multi_instances.worker_client import WorkerClient
from models.vllm_model import sampling_params_from_generation_params
from models.vllm_response_wrapper import VLLMResponseWrapper


class RemoteVLLMModel:
    def __init__(
        self,
        worker_client: "WorkerClient",
        use_chat_template: bool = True,
        reasoning_parser_mode=None,
        chat_template=None,
    ):
        self.worker_client = worker_client
        self.use_chat_template = use_chat_template
        self.reasoning_parser_mode = reasoning_parser_mode
        self.chat_template = chat_template

    async def query(self, prompt, generation_params):
        sampling_params = sampling_params_from_generation_params(generation_params)

        response = await self.worker_client.call(
            "generate" if not self.use_chat_template else "chat",
            prompt,
            sampling_params=sampling_params,
            chat_template=self.chat_template,
        )
        return VLLMResponseWrapper(
            response,
            n_logprobs=getattr(generation_params, "logprobs", None),
            reasoning_parser_mode=self.reasoning_parser_mode,
        )
