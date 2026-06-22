from __future__ import annotations

from ...multi_vllm_instances.worker_client import WorkerClient


class RemoteEncoderModel:
    def __init__(self, worker_client: "WorkerClient"):
        self.worker_client = worker_client

    async def encode(self, pairs, batch_size=8):
        """
        pairs: list of (text1, text2)
        returns: logits list (len=pairs)
        """
        return await self.worker_client.call(
            "encode_full", pairs, batch_size=batch_size
        )
