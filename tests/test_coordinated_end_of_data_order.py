"""EndOfData ordering for nodes with max_tasks > 1.

coordinated_end_of_data holds EndOfData until no task of a node is still working on an
item. Items that a node emits without reaching that layer (cached results returned by
data_cache) were not counted, so EndOfData could overtake them in a full downstream
queue and the downstream batching layer rejected them ("Items were pushed to the
batching layer after EndOfData signal was already seen").

The graphs below use small queues so that the tasks of the parallel node block while
handing items downstream. asyncio scheduling is deterministic, so without the fix the
reproduction cases fail on every run.
"""

from collections import defaultdict

import pytest

from async_graph_bench import BenchmarkManager, JSONDataStore, NodeConfig, benchmark_run
from async_graph_bench.execution_nodes import NodeExecutionWrapper
from async_graph_bench.sampling_config import SamplingConfig
from async_graph_bench.utils.end_of_data import EndOfData
from tests.fixtures import MockNode, SimpleMockDataSource

EOD = "EndOfData"


@pytest.fixture
def node_events(monkeypatch):
    """Records, per node object, the item keys and EndOfData signals its node wrapper receives."""
    events = defaultdict(list)

    class RecordingNodeExecutionWrapper(NodeExecutionWrapper):
        async def execute(self, items, *args, **kwargs):
            if isinstance(items, EndOfData):
                events[self.node].append(EOD)
            else:
                for item in items if isinstance(items, list) else [items]:
                    events[self.node].append((item["id"], item.get("iter", 0)))
            async for out in super().execute(items, *args, **kwargs):
                yield out

    monkeypatch.setattr(
        benchmark_run, "NodeExecutionWrapper", RecordingNodeExecutionWrapper
    )
    return events


def upstream(node, max_tasks):
    return NodeConfig(
        node, id="U", data_store=JSONDataStore, max_tasks=max_tasks, greedy=True
    )


def fill_upstream_cache(path, n_items, max_tasks=1, iterations=1):
    """First run: compute and store the upstream node for the first n_items ids."""
    BenchmarkManager(
        SimpleMockDataSource(n_items),
        [upstream(MockNode(["value"], ["u"]), max_tasks)],
        str(path),
        show_progress_bars=False,
        iterations=iterations,
    ).run_benchmark()


def run(path, n_items, nodes, iterations=1, iterations_first=None, progress=False):
    manager = BenchmarkManager(
        SimpleMockDataSource(n_items),
        nodes,
        str(path),
        show_progress_bars=progress,
        halt_on_exception=False,
        raise_exceptions=False,
        iterations=iterations,
        iterations_first=iterations_first,
    )
    manager.run_benchmark()
    exceptions = [repr(e.exception) for r in manager.runs for e in r.exceptions]
    return manager, exceptions


def stored_keys(manager, node_id):
    return sorted(
        (d["id"], d.get("iter", 0)) for d in manager.store_per_node[node_id].data
    )


def assert_node_events(events, expected_keys):
    """Every key exactly once, then exactly one EndOfData as the last signal."""
    assert events.count(EOD) == 1
    assert events[-1] == EOD
    assert sorted(events[:-1]) == sorted(expected_keys)


def all_keys(n_items, iterations=1):
    return [(i, it) for i in range(n_items) for it in range(iterations)]


@pytest.mark.parametrize(
    "n_cached, max_tasks",
    [(10, 4), (10, 2), (5, 2)],
    ids=["fully-cached-4-tasks", "fully-cached-2-tasks", "partly-cached-2-tasks"],
)
def test_cached_parallel_node_feeding_batched_node(
    tmp_path, node_events, n_cached, max_tasks
):
    """Reproduction: cached items must reach the batched child before EndOfData."""
    n = 10
    fill_upstream_cache(tmp_path, n_cached)
    node_events.clear()
    child = MockNode(["u"], ["d"])
    manager, exceptions = run(
        tmp_path,
        n,
        [
            upstream(MockNode(["value"], ["u"]), max_tasks),
            NodeConfig(
                child,
                id="D",
                data_store=JSONDataStore,
                batch_size=3,  # 10 items: final partial batch of 1
                queue_size=2,
                greedy=True,
            ),
        ],
    )
    assert exceptions == []
    assert stored_keys(manager, "D") == all_keys(n)
    assert_node_events(node_events[child], all_keys(n))


def test_parallel_batched_node_flushes_final_partial_batch(tmp_path, node_events):
    """A batched node with max_tasks > 1 still sends its last partial batch before EndOfData."""
    n = 10
    parent = MockNode(["value"], ["u"])
    child = MockNode(["u"], ["d"])
    manager, exceptions = run(
        tmp_path,
        n,
        [
            NodeConfig(
                parent,
                id="U",
                data_store=JSONDataStore,
                batch_size=3,
                max_tasks=3,
                queue_size=2,
                greedy=True,
            ),
            NodeConfig(
                child,
                id="D",
                data_store=JSONDataStore,
                batch_size=4,
                max_tasks=2,
                queue_size=2,
                greedy=True,
            ),
        ],
    )
    assert exceptions == []
    assert stored_keys(manager, "U") == all_keys(n)
    assert stored_keys(manager, "D") == all_keys(n)
    assert_node_events(node_events[parent], all_keys(n))
    assert_node_events(node_events[child], all_keys(n))


def test_cached_node_with_single_task_unchanged(tmp_path, node_events):
    """max_tasks == 1 has no coordination layer and was already correct."""
    n = 10
    fill_upstream_cache(tmp_path, n)
    node_events.clear()
    child = MockNode(["u"], ["d"])
    manager, exceptions = run(
        tmp_path,
        n,
        [
            upstream(MockNode(["value"], ["u"]), 1),
            NodeConfig(
                child,
                id="D",
                data_store=JSONDataStore,
                batch_size=3,
                queue_size=2,
                greedy=True,
            ),
        ],
    )
    assert exceptions == []
    assert stored_keys(manager, "D") == all_keys(n)
    assert_node_events(node_events[child], all_keys(n))


def test_multi_parent_parallel_node_gets_one_end_of_data(tmp_path, node_events):
    """A node with two parallel parents and max_tasks > 1 receives exactly one EndOfData."""
    n = 10
    fill_upstream_cache(tmp_path, 6)
    node_events.clear()
    child = MockNode(["u", "w"], ["d"])
    manager, exceptions = run(
        tmp_path,
        n,
        [
            upstream(MockNode(["value"], ["u"]), 3),
            NodeConfig(
                MockNode(["value"], ["w"]),
                id="W",
                data_store=JSONDataStore,
                max_tasks=3,
                batch_size=2,
                greedy=True,
            ),
            NodeConfig(
                child,
                id="D",
                data_store=JSONDataStore,
                batch_size=3,
                max_tasks=3,
                queue_size=2,
                greedy=True,
            ),
        ],
    )
    assert exceptions == []
    assert stored_keys(manager, "D") == all_keys(n)
    assert_node_events(node_events[child], all_keys(n))


def test_parallel_node_with_skipped_items(tmp_path, node_events):
    """Skipped items are inside the coordination layer; EndOfData still arrives exactly once."""
    n = 10
    leaf_1 = MockNode(["u"], ["d1"])
    leaf_2 = MockNode(["u"], ["d2"])

    def nodes(n1, n2):
        return [
            upstream(MockNode(["value"], ["u"]), 3),
            NodeConfig(
                n1,
                id="D1",
                data_store=JSONDataStore,
                batch_size=2,
                max_tasks=3,
                queue_size=2,
                greedy=True,
            ),
            NodeConfig(
                n2,
                id="D2",
                data_store=JSONDataStore,
                batch_size=2,
                max_tasks=3,
                queue_size=2,
                greedy=True,
            ),
        ]

    # First run: D1 complete for the first 6 ids, D2 not computed at all.
    BenchmarkManager(
        SimpleMockDataSource(6),
        nodes(MockNode(["u"], ["d1"]), MockNode(["u"], ["d2"]))[:2],
        str(tmp_path),
        show_progress_bars=False,
    ).run_benchmark()
    node_events.clear()

    manager, exceptions = run(tmp_path, n, nodes(leaf_1, leaf_2))
    assert exceptions == []
    assert stored_keys(manager, "D1") == all_keys(n)
    assert stored_keys(manager, "D2") == all_keys(n)
    assert_node_events(node_events[leaf_1], all_keys(n)[6:])
    assert_node_events(node_events[leaf_2], all_keys(n))


@pytest.mark.parametrize("iterations_first", [True, False])
def test_cached_parallel_node_with_iterations(tmp_path, node_events, iterations_first):
    n, iterations = 6, 2
    fill_upstream_cache(tmp_path, n, iterations=iterations)
    node_events.clear()
    child = MockNode(["u"], ["d"])
    manager, exceptions = run(
        tmp_path,
        n,
        [
            upstream(MockNode(["value"], ["u"]), 4),
            NodeConfig(
                child,
                id="D",
                data_store=JSONDataStore,
                batch_size=5,
                queue_size=2,
                greedy=True,
            ),
        ],
        iterations=iterations,
        iterations_first=iterations_first,
    )
    assert exceptions == []
    assert stored_keys(manager, "D") == all_keys(n, iterations)
    assert_node_events(node_events[child], all_keys(n, iterations))


class SampledSum:
    def __init__(self):
        self.requires = ["sampled_u"]
        self.provides = ["s"]

    def __call__(self, item_stats, **kwargs):
        return {"s": [sum(values) for values in item_stats["sampled_u"]]}


def test_cached_parallel_node_feeding_sampling_node(tmp_path, node_events):
    """A parallel sampling node behind a cached parallel node gets every group once."""
    n, iterations = 6, 2
    fill_upstream_cache(tmp_path, n, iterations=iterations)
    node_events.clear()
    sampler = SampledSum()
    manager, exceptions = run(
        tmp_path,
        n,
        [
            upstream(MockNode(["value"], ["u"]), 3),
            NodeConfig(
                sampler,
                id="S",
                data_store=JSONDataStore,
                batch_size=2,
                max_tasks=2,
                queue_size=2,
                greedy=True,
                sampling_config=SamplingConfig(sampling_size=iterations),
            ),
        ],
        iterations=iterations,
    )
    assert exceptions == []
    # mode "first": one result per id, for the group's first iteration
    expected = [(i, 0) for i in range(n)]
    assert stored_keys(manager, "S") == expected
    assert_node_events(node_events[sampler], expected)


def test_progress_bars_complete(tmp_path):
    """Progress bars count every item once with the coordination layer outside them."""
    n = 10
    fill_upstream_cache(tmp_path, 5)
    manager, exceptions = run(
        tmp_path,
        n,
        [
            upstream(MockNode(["value"], ["u"]), 3),
            NodeConfig(
                MockNode(["u"], ["d"]),
                id="D",
                data_store=JSONDataStore,
                batch_size=3,
                max_tasks=2,
                queue_size=2,
                greedy=True,
            ),
        ],
        progress=True,
    )
    assert exceptions == []
    bars = [bar for r in manager.runs for bar in r._progress_bars]
    assert len(bars) == 2
    assert all(bar.n == bar.total == n for bar in bars)
