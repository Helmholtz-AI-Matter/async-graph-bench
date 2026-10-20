"""Skipping already computed items must not depend on the order in which a DataSource yields its items.

The manager indexes items by their position in ``iter_ids()``; skip tables use these positions. ``_idx`` is
therefore looked up by id (``item_index``) instead of counted in ``iter_items()`` order.
"""

import pytest

from async_graph_bench import BenchmarkManager, DiskCacheStore, NodeConfig
from async_graph_bench.execution_nodes.data_source_execution_wrapper import (
    DataSourceExecutionWrapper,
)
from async_graph_bench.utils.end_of_data import EndOfData
from async_graph_bench.utils.helpers import build_combined_keys
from tests.fixtures import MockNode, SimpleMockDataSource


class ReorderedIdsSource(SimpleMockDataSource):
    """iter_items() yields 0, 1, 2; iter_ids() yields the same ids in another order (1, 2, 0)."""

    def iter_ids(self):
        yield from [1, 2, 0]


def index_for(ids, iterations=1, iterations_first=True):
    keys = build_combined_keys(
        ids, iterations=iterations, iterations_first=iterations_first
    )
    return {key: i for i, key in enumerate(keys)}


async def collect(wrapper):
    return [
        i
        for i in [item async for item in wrapper.execute()]
        if not isinstance(i, EndOfData)
    ]


@pytest.mark.asyncio
async def test_idx_from_index_independent_of_order():
    ds = ReorderedIdsSource(3)
    index = index_for(ds.iter_ids())
    items = await collect(DataSourceExecutionWrapper(ds.iter_items, item_index=index))
    assert {item["id"]: item["_idx"] for item in items} == {1: 0, 2: 1, 0: 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("iterations_first", [True, False])
async def test_idx_from_index_with_iterations(iterations_first):
    ds = ReorderedIdsSource(3)
    index = index_for(ds.iter_ids(), iterations=2, iterations_first=iterations_first)
    wrapper = DataSourceExecutionWrapper(
        ds.iter_items, iterations=2, iterations_first=iterations_first, item_index=index
    )
    items = await collect(wrapper)
    assert len(items) == 6
    assert all(item["_idx"] == index[(item["iter"], item["id"])] for item in items)


@pytest.mark.asyncio
async def test_without_index_idx_is_counted():
    items = await collect(DataSourceExecutionWrapper(ReorderedIdsSource(3).iter_items))
    assert [item["_idx"] for item in items] == [0, 1, 2]


@pytest.mark.asyncio
async def test_item_not_in_iter_ids_raises():
    wrapper = DataSourceExecutionWrapper(
        SimpleMockDataSource(3).iter_items, item_index=index_for([0, 1])
    )
    with pytest.raises(KeyError, match="not part of iter_ids"):
        await collect(wrapper)


def test_rerun_computes_missing_item_despite_reordered_ids(tmp_path):
    """Items 0 and 1 stored, item 2 missing: the rerun must compute exactly item 2 (before the fix it was skipped
    and an already stored item was re-sent instead)."""
    node = MockNode(requires=["value"], provides=["doubled"], multiply=2.0)

    def run():
        mgr = BenchmarkManager(
            data_source=ReorderedIdsSource(3),
            nodes=[
                NodeConfig(
                    node,
                    id="double",
                    greedy=True,
                    data_store=DiskCacheStore,
                    batch_size=1,
                )
            ],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
        )
        mgr.run_benchmark()
        return mgr

    mgr = run()
    store = mgr.store_per_node["double"]
    store.delete(2)
    store.cache.close()
    node.call_count = 0

    mgr = run()
    store = mgr.store_per_node["double"]
    assert node.call_count == 1
    assert store.load(2)["doubled"] == pytest.approx(4.0)
    store.cache.close()
