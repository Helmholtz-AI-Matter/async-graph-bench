import pytest
import asyncio
from typing import List, Dict, Any
from async_graph_bench.execution_nodes.node_execution_wrapper import NodeExecutionWrapper
from async_graph_bench.utils.end_of_data import EndOfData
from async_graph_bench import NodeConfig
from tests.fixtures import MockNode


class TestNodeExecutionWrapperIntermediate:
    @pytest.mark.asyncio
    async def test_forward_provides(self):
        node = MockNode(requires=["text"], provides=["processed"], multiply=10)
        wrapper = NodeExecutionWrapper(node)

        items = [
            {"id": 1, "text": 2, "other": "keep"},
            {"id": 2, "text": 3, "other": "keep2"},
        ]

        collected = []
        async for out in wrapper.execute(items):
            collected.append(out)

        assert len(collected) == 2
        assert collected[0]["processed"] == pytest.approx(20)
        assert collected[0]["other"] == "keep"

    @pytest.mark.asyncio
    async def test_single_item(self):
        node = MockNode(requires=["value"], provides=["result"], multiply=5)
        wrapper = NodeExecutionWrapper(node)

        collected = []
        async for out in wrapper.execute({"id": 1, "value": 10}):
            collected.append(out)

        assert len(collected) == 1
        assert collected[0]["result"] == 50

    @pytest.mark.asyncio
    async def test_eyecall_async_call(self):
        async_node = MockNode(requires=["val"], provides=["out"], async_call=True, multiply=2)
        wrapper = NodeExecutionWrapper(async_node)

        collected = []
        async for out in wrapper.execute([{"id": 1, "val": 5}]):
            collected.append(out)

        assert len(collected) == 1
        assert collected[0]["out"] == 10


class TestNodeExecutionWrapperConsumer:
    @pytest.mark.asyncio
    async def test_consumer_dict_of_lists(self):
        node = MockNode(requires=["text"])
        wrapper = NodeExecutionWrapper(node)

        items = [
            {"id": 1, "text": 10},
            {"id": 2, "text": 20},
        ]

        collected = []
        async for out in wrapper.execute(items):
            collected.append(out)

        assert len(collected) == 2
        assert collected[0]["id"] == 1
        assert collected[1]["id"] == 2

    @pytest.mark.asyncio
    async def test_consumer_flat_list(self):
        node = MockNode(requires=["val"])
        wrapper = NodeExecutionWrapper(node)

        items = [
            {"id": 1, "val": 10},
            {"id": 2, "val": 20},
        ]

        collected = []
        async for out in wrapper.execute(items):
            collected.append(out)

        assert len(collected) == 2
        assert collected[0]["id"] == 1


class TestNodeExecutionWrapperEndOfData:
    @pytest.mark.asyncio
    async def test_eod_passthrough(self):
        node = MockNode(requires=["text"], provides=["out"])
        wrapper = NodeExecutionWrapper(node)

        collected = []
        async for out in wrapper.execute(EndOfData()):
            collected.append(out)

        assert len(collected) == 1
        assert isinstance(collected[0], EndOfData)