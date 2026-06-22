import pytest
from async_graph_bench.utils.end_of_data import EndOfData
from async_graph_bench.execution_nodes.decorators import (
    batching,
    skip_indices,
    coordinated_end_of_data,
    multi_incoming_node,
    progress_wrapper,
    with_resources,
)
from async_graph_bench.utils.resource_pool import ResourcePool
from bitarray import bitarray


async def collect_items(gen_func):
    items = []
    async for item in gen_func(None):
        items.append(item)
    return items


async def simple_gen(items):
    for item in items:
        yield item


async def single_item_gen(item):
    yield item


class TestBatching:
    @pytest.mark.asyncio
    async def test_exact_batch_size(self):
        batched = batching(single_item_gen, batch_size=2)

        emitted = []
        async for item in batched({"id": 1}):
            emitted.append(item)
        async for item in batched({"id": 2}):
            emitted.append(item)
        end_result = []
        async for item in batched(EndOfData()):
            end_result.append(item)

        assert len(emitted) == 1
        assert isinstance(end_result[0], EndOfData)

    @pytest.mark.asyncio
    async def test_batch_emits_on_size(self):
        received_batches = []

        async def inner_gen(item):
            received_batches.append(item)
            for x in item:
                yield x

        batched = batching(inner_gen, batch_size=3)

        async for _ in batched({"v": 1}):
            pass
        async for _ in batched({"v": 2}):
            pass
        result_last = batched({"v": 3})

        collected = []
        async for item in result_last:
            collected.append(item)

        assert any(b == [{"v": 1}, {"v": 2}, {"v": 3}] for b in received_batches)

    @pytest.mark.asyncio
    async def test_flush_remainder_on_eod(self):
        received_batches = []

        async def inner_gen(item):
            if not isinstance(item, EndOfData):
                received_batches.append(item)
            else:
                yield item

        batched = batching(inner_gen, batch_size=5)

        async for _ in batched({"v": 1}):
            pass
        async for _ in batched({"v": 2}):
            pass

        end_collected = []
        async for item in batched(EndOfData()):
            end_collected.append(item)

        assert any(b == [{"v": 1}, {"v": 2}] for b in received_batches)

    @pytest.mark.asyncio
    async def test_empty_batch_on_eod(self):
        eod_received = False

        async def inner_gen(item):
            nonlocal eod_received
            if isinstance(item, EndOfData):
                eod_received = True
                yield item

        batched = batching(inner_gen, batch_size=3)
        async for _ in batched(EndOfData()):
            pass
        assert eod_received


class TestSkipIndices:
    @pytest.mark.asyncio
    async def test_skip_marked_indices(self):
        skip = bitarray("1010")
        passed_through = []

        async def inner_gen(item):
            passed_through.append(item)
            yield item

        wrapped = skip_indices(inner_gen, skip)

        for item in [
            {"_idx": 0, "v": 0},
            {"_idx": 1, "v": 1},
            {"_idx": 2, "v": 2},
            {"_idx": 3, "v": 3},
        ]:
            async for _ in wrapped(item):
                pass

        idxs = [p["_idx"] for p in passed_through]
        assert 0 not in idxs
        assert 1 in idxs
        assert 2 not in idxs
        assert 3 in idxs

    @pytest.mark.asyncio
    async def test_eod_passes_through(self):
        skip = bitarray("1111")
        eod_seen = False

        async def inner_gen(item):
            nonlocal eod_seen
            if isinstance(item, EndOfData):
                eod_seen = True
                yield item

        wrapped = skip_indices(inner_gen, skip)
        async for _ in wrapped(EndOfData()):
            pass
        assert eod_seen


class TestCoordinatedEndOfData:
    @pytest.mark.asyncio
    async def test_forwards_data(self):
        forwarded = []

        async def inner_gen(item):
            if not isinstance(item, EndOfData):
                forwarded.append(item)
                yield item

        coord = coordinated_end_of_data(inner_gen, name="test")
        async for out in coord({"v": 1}):
            pass
        async for _ in coord(EndOfData()):
            pass
        assert len(forwarded) == 1


class TestMultiIncomingNode:
    @pytest.mark.asyncio
    async def test_waits_for_count(self):
        processed = []

        async def inner_gen(item):
            processed.append(item)
            yield item

        wrapper = multi_incoming_node(inner_gen, incoming_nodes_count=2)

        for i in range(2):
            async for _ in wrapper({"id": 1, "iter": 0, "src": i}):
                pass

        assert len(processed) == 1

    @pytest.mark.asyncio
    async def test_multiple_ids(self):
        processed = []

        async def inner_gen(item):
            processed.append(item)
            yield item

        wrapper = multi_incoming_node(inner_gen, incoming_nodes_count=2)

        for src in range(2):
            async for _ in wrapper({"id": 1, "iter": 0, "src": src}):
                pass
        for src in range(2):
            async for _ in wrapper({"id": 2, "iter": 0, "src": src}):
                pass

        assert len(processed) == 2


class TestWithResources:
    @pytest.mark.asyncio
    async def test_single_pool(self):
        resources_used = []

        async def inner_gen(item, resource):
            resources_used.append(resource)
            yield {"res": resource}

        pool = ResourcePool(["r1", "r2"])
        wrapped = with_resources(inner_gen, [pool])

        collected = []
        async for out in wrapped({"id": 1}):
            collected.append(out)

        assert len(collected) == 1
        assert len(resources_used) == 1

    @pytest.mark.asyncio
    async def test_eod_passes_through_single_pool(self):
        eod_received = False

        async def inner_gen(item, resource=None):
            nonlocal eod_received
            if isinstance(item, EndOfData):
                eod_received = True
                yield item

        pool = ResourcePool(["r1"])
        wrapped = with_resources(inner_gen, [pool])
        async for _ in wrapped(EndOfData()):
            pass
        assert eod_received


class TestProgressWrapper:
    @pytest.mark.asyncio
    async def test_increments_on_items(self):
        counts = [0]

        class MockBar:
            def update(self, n=1):
                counts[0] += n

        bar = MockBar()

        async def inner_gen(item):
            yield item

        wrapped = progress_wrapper(inner_gen, bar)
        async for _ in wrapped({"id": 1}):
            pass
        async for _ in wrapped({"id": 2}):
            pass

        assert counts[0] == 2
