import pytest
from async_graph_bench.execution_nodes.data_source_execution_wrapper import (
    DataSourceExecutionWrapper,
    _ensure_async,
)
from async_graph_bench.utils.end_of_data import EndOfData
from tests.fixtures import SimpleMockDataSource, AsyncMockDataSource


class TestEnsureAsync:
    @pytest.mark.asyncio
    async def test_already_async(self):
        async def async_iter():
            yield 1
            yield 2

        source = await _ensure_async(async_iter())
        items = []
        async for item in source:
            items.append(item)
        assert items == [1, 2]

    @pytest.mark.asyncio
    async def test_sync_iterable(self):
        source = await _ensure_async(iter([1, 2, 3]))
        items = []
        async for item in source:
            items.append(item)
        assert items == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_non_iterable_raises(self):
        with pytest.raises(TypeError):
            await _ensure_async(42)


class TestDataSourceExecutionWrapper:
    @pytest.mark.asyncio
    async def test_single_iteration_no_iter_field(self):
        ds = SimpleMockDataSource(3)
        wrapper = DataSourceExecutionWrapper(ds.iter_items, iterations=1, iterations_first=True)

        items = []
        async for item in wrapper.execute():
            items.append(item)

        data_items = [item for item in items if not isinstance(item, EndOfData)]
        assert len(data_items) == 3
        for item in data_items:
            assert "_idx" in item
            assert "iter" not in item

    @pytest.mark.asyncio
    async def test_iterations_first_true(self):
        ds = SimpleMockDataSource(2)
        wrapper = DataSourceExecutionWrapper(ds.iter_items, iterations=3, iterations_first=True)

        items = []
        async for item in wrapper.execute():
            items.append(item)

        data_items = [item for item in items if not isinstance(item, EndOfData)]
        assert len(data_items) == 6
        iters = [item["iter"] for item in data_items]
        assert iters == [0, 1, 2, 0, 1, 2]

    @pytest.mark.asyncio
    async def test_iterations_first_false(self):
        ds = SimpleMockDataSource(2)
        wrapper = DataSourceExecutionWrapper(ds.iter_items, iterations=3, iterations_first=False)

        items = []
        async for item in wrapper.execute():
            items.append(item)

        data_items = [item for item in items if not isinstance(item, EndOfData)]
        assert len(data_items) == 6
        iters = [item["iter"] for item in data_items]
        assert iters == [0, 0, 1, 1, 2, 2]

    @pytest.mark.asyncio
    async def test_ends_with_eod(self):
        ds = SimpleMockDataSource(2)
        wrapper = DataSourceExecutionWrapper(ds.iter_items, iterations=1)

        items = []
        async for item in wrapper.execute():
            items.append(item)

        assert isinstance(items[-1], EndOfData)

    @pytest.mark.asyncio
    async def test_unique_idx(self):
        ds = SimpleMockDataSource(3)
        wrapper = DataSourceExecutionWrapper(ds.iter_items, iterations=2, iterations_first=True)

        items = []
        async for item in wrapper.execute():
            items.append(item)

        non_eod = [item for item in items if not isinstance(item, EndOfData)]
        idxs = [item["_idx"] for item in non_eod]
        assert len(idxs) == len(set(idxs))

    @pytest.mark.asyncio
    async def test_async_data_source(self):
        ds = AsyncMockDataSource(3)
        wrapper = DataSourceExecutionWrapper(ds.iter_items, iterations=1)

        items = []
        async for item in wrapper.execute():
            items.append(item)

        non_eod = [item for item in items if not isinstance(item, EndOfData)]
        assert len(non_eod) == 3