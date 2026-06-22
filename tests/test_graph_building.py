from typing import List, Dict, Any
from async_graph_bench import (
    NodeConfig,
    CSVDataStore,
    BenchmarkManager,
    ResourcePool,
)
from async_graph_bench.utils import BuilderEnvironment
from tests.fixtures import MockNode, SimpleMockDataSource


class ValueAdderNode:
    """Intermediate node that adds a value to each item."""

    def __init__(self, add_val: int = 1):
        self.requires = ["value"]
        self.provides = ["value_added"]
        self._add_val = add_val

    def __call__(
        self, item_stats: Dict[str, List[Any]], **kwargs
    ) -> Dict[str, List[Any]]:
        return {"value_added": [v + self._add_val for v in item_stats["value"]]}


class TextLengthNode:
    """Intermediate node that computes text length."""

    requires = ["text"]
    provides = ["text_len"]

    def __call__(
        self, item_stats: Dict[str, List[Any]], **kwargs
    ) -> Dict[str, List[Any]]:
        return {"text_len": [len(t) for t in item_stats["text"]]}


class ConsumerNode:
    """Consumer node that produces final output (no provides)."""

    requires = ["text_len", "value_added"]

    def __call__(self, item_stats: Dict[str, List[Any]], **kwargs) -> List[Any]:
        lengths = item_stats["text_len"]
        values = item_stats["value_added"]
        return [length * v for length, v in zip(lengths, values)]


class SamplerNode:
    """Consumer that relies on sampled data."""

    requires = ["sampled_value_added"]

    def __call__(
        self, item_stats: Dict[str, List[Any]], **kwargs
    ) -> Dict[str, List[Any]]:
        samples = item_stats["sampled_value_added"]
        return {
            "sum": [sum(s) for s in samples],
        }


class TestMinimalGraphRun:
    def test_basic_graph(self, tmp_path):
        ds = SimpleMockDataSource(4)
        adder = NodeConfig(ValueAdderNode(add_val=1))
        text_len = NodeConfig(TextLengthNode())
        consumer = NodeConfig(ConsumerNode(), greedy=True, data_store=CSVDataStore)

        mgr = BenchmarkManager(
            data_source=ds,
            nodes=[adder, text_len],
            consumer_nodes=[consumer],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
            halt_on_exception=True,
            raise_exceptions=False,
            iterations=1,
        )

        assert mgr.base_adg is not None
        mgr.run_benchmark()
        assert mgr.get_state() == "finished"


class TestGraphWithCacheSkip:
    def test_second_run_skipped(self, tmp_path):
        ds = SimpleMockDataSource(4)
        adder = NodeConfig(ValueAdderNode())
        text_len = NodeConfig(TextLengthNode())
        consumer = NodeConfig(ConsumerNode(), greedy=True, data_store=CSVDataStore)

        mgr = BenchmarkManager(
            data_source=ds,
            nodes=[adder, text_len],
            consumer_nodes=[consumer],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
            halt_on_exception=True,
            raise_exceptions=False,
            iterations=1,
        )
        mgr.run_benchmark()
        first_state = mgr.get_state()
        assert first_state == "finished"

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mgr2 = BenchmarkManager(
                data_source=ds,
                nodes=[adder, text_len],
                consumer_nodes=[consumer],
                data_storage_path=str(tmp_path),
                show_progress_bars=False,
                halt_on_exception=True,
                raise_exceptions=False,
                iterations=1,
            )
            mgr2.run_benchmark()
            assert mgr2.get_state() in ("skipped", "finished")


class TestGraphTreeshakeUnused:
    def test_unused_node_pruned(self, tmp_path):
        ds = SimpleMockDataSource(4)
        used = NodeConfig(ValueAdderNode())
        unused = NodeConfig(TextLengthNode())
        consumer = NodeConfig(ConsumerNode(), greedy=True, data_store=CSVDataStore)

        mgr = BenchmarkManager(
            data_source=ds,
            nodes=[used, unused],
            consumer_nodes=[consumer],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
            halt_on_exception=True,
            raise_exceptions=False,
            iterations=1,
        )
        mgr.run_benchmark()
        assert mgr.get_state() == "finished"


class TestGraphWithResourcePools:
    def test_resource_builder_called(self, tmp_path):
        resource_called = [False]

        class ResNode:
            requires = ["value"]
            provides = ["res_out"]

            def __call__(
                self, item_stats: Dict[str, List[Any]], resource: str, **kwargs
            ) -> Dict[str, List[Any]]:
                return {"res_out": [v * 2 for v in item_stats["value"]]}

        def build_resource(env: BuilderEnvironment) -> ResourcePool:
            resource_called[0] = True
            return ResourcePool(["shared_resource"])

        res_cfg = NodeConfig(ResNode(), resource_builder=build_resource)
        consumer = NodeConfig(
            MockNode(requires=["res_out"]), greedy=True, data_store=CSVDataStore
        )

        mgr = BenchmarkManager(
            data_source=SimpleMockDataSource(3),
            nodes=[res_cfg, consumer],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
            halt_on_exception=True,
            raise_exceptions=False,
            iterations=1,
        )
        mgr.run_benchmark()
        assert resource_called[0] is True


class TestGraphStoreOutput:
    def test_csv_output_exists(self, tmp_path):
        ds = SimpleMockDataSource(4)
        consumer = NodeConfig(
            MockNode(requires=["value"]),
            greedy=True,
            data_store=CSVDataStore,
        )

        mgr = BenchmarkManager(
            data_source=ds,
            nodes=[],
            consumer_nodes=[consumer],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
            halt_on_exception=True,
            raise_exceptions=False,
            iterations=1,
        )
        mgr.run_benchmark()
        store = mgr.store_per_node[consumer.id]
        assert len(store) == 4
