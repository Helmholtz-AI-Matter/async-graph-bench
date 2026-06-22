import time
from typing import List, Dict, Any
from async_graph_bench import (
    NodeConfig,
    CSVDataStore,
    BenchmarkManager,
    ResourcePool,
)
from async_graph_bench.utils import BuilderEnvironment
from tests.fixtures import MockNode, SimpleMockDataSource


class SlowComputeNode:
    """Node that sleeps to simulate work, using a resource pool."""

    requires = ["value"]
    provides = ["computed"]

    def __init__(self, delay: float = 0.1):
        self.delay = delay

    def __call__(
        self, item_stats: Dict[str, List[Any]], resource: str, **kwargs
    ) -> Dict[str, List[Any]]:
        time.sleep(self.delay)
        return {"computed": [v * 2 for v in item_stats["value"]]}


class SlowValueDoubler:
    """Simple slow node adding value_doubled."""

    requires = ["value"]
    provides = ["value_doubled"]

    def __call__(
        self, item_stats: Dict[str, List[Any]], resource: str, **kwargs
    ) -> Dict[str, List[Any]]:
        time.sleep(0.1)
        return {"value_doubled": [v * 2 for v in item_stats["value"]]}


class SlowTextAnalyzer:
    """Simple slow node adding text_len."""

    requires = ["text"]
    provides = ["text_len"]

    def __call__(
        self, item_stats: Dict[str, List[Any]], resource: str, **kwargs
    ) -> Dict[str, List[Any]]:
        time.sleep(0.1)
        return {"text_len": [len(t) for t in item_stats["text"]]}


class SlowConsumerNode:
    """Consumer that requires both outputs."""

    requires = ["value_doubled", "text_len"]

    def __call__(
        self, item_stats: Dict[str, List[Any]], resource: str, **kwargs
    ) -> List[Any]:
        return [
            v * lt for v, lt in zip(item_stats["value_doubled"], item_stats["text_len"])
        ]


def build_resource_pool(env: BuilderEnvironment, pool_size: int = 2) -> ResourcePool:
    return ResourcePool([f"worker_{i}" for i in range(pool_size)])


class TestConcurrentExecution:
    def test_concurrent_nodes_faster_than_sequential(self, tmp_path):
        ds = SimpleMockDataSource(4)

        doubler = NodeConfig(
            SlowValueDoubler(),
            resource_builder=lambda env: build_resource_pool(env, 4),
            max_tasks=4,
            batch_size=4,
        )
        analyzer = NodeConfig(
            SlowTextAnalyzer(),
            resource_builder=lambda env: build_resource_pool(env, 4),
            max_tasks=4,
            batch_size=4,
        )
        consumer = NodeConfig(
            SlowConsumerNode(),
            resource_builder=lambda env: build_resource_pool(env, 2),
            max_tasks=2,
            greedy=True,
            data_store=CSVDataStore,
            batch_size=4,
        )

        mgr = BenchmarkManager(
            data_source=ds,
            nodes=[doubler, analyzer],
            consumer_nodes=[consumer],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
            halt_on_exception=True,
            raise_exceptions=False,
            iterations=1,
        )

        mgr.run_benchmark()
        assert mgr.get_state() == "finished"

    def test_multiple_pools_no_deadlock(self, tmp_path):
        ds = SimpleMockDataSource(3)

        def build_pool_a(env: BuilderEnvironment) -> ResourcePool:
            return ResourcePool(["a1", "a2"])

        def build_pool_b(env: BuilderEnvironment) -> ResourcePool:
            return ResourcePool(["b1", "b2"])

        class MultiResourceNode:
            requires = ["value"]
            provides = ["multi_out"]

            def __call__(
                self,
                item_stats: Dict[str, List[Any]],
                resource_a: str,
                resource_b: str,
                **kwargs,
            ) -> Dict[str, List[Any]]:
                return {"multi_out": [v * 3 for v in item_stats["value"]]}

        multi_node = NodeConfig(
            MultiResourceNode(),
            resource_builder=lambda env: [build_pool_a(env), build_pool_b(env)],
            max_tasks=2,
        )
        consumer = NodeConfig(
            MockNode(requires=["multi_out"]),
            greedy=True,
            data_store=CSVDataStore,
        )

        mgr = BenchmarkManager(
            data_source=ds,
            nodes=[multi_node, consumer],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
            halt_on_exception=True,
            raise_exceptions=False,
            iterations=1,
        )
        mgr.run_benchmark()
        assert mgr.get_state() == "finished"

    def test_batching_with_concurrent_inputs(self, tmp_path):
        ds = SimpleMockDataSource(8)

        adder = NodeConfig(
            SlowValueDoubler(),
            resource_builder=lambda env: build_resource_pool(env, 2),
            max_tasks=2,
            batch_size=4,
        )
        consumer = NodeConfig(
            MockNode(requires=["value_doubled"]),
            greedy=True,
            data_store=CSVDataStore,
            batch_size=4,
        )

        mgr = BenchmarkManager(
            data_source=ds,
            nodes=[adder, consumer],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
            halt_on_exception=True,
            raise_exceptions=False,
            iterations=1,
        )
        mgr.run_benchmark()
        assert mgr.get_state() == "finished"
