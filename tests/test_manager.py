import pytest
from async_graph_bench import (
    BenchmarkManager,
    NodeConfig,
    CSVDataStore,
)
from tests.fixtures import MockNode, SimpleMockDataSource


class TestBenchmarkManagerInit:
    def test_duplicate_node_ids_raises(self, tmp_path):
        ds = SimpleMockDataSource(4)
        node = MockNode(requires=["text"])
        cfg1 = NodeConfig(node, id="dup", greedy=True, data_store=CSVDataStore)
        cfg2 = NodeConfig(node, id="dup", greedy=True, data_store=CSVDataStore)

        with pytest.raises(Exception, match="Duplicate string found"):
            BenchmarkManager(
                data_source=ds,
                nodes=[],
                consumer_nodes=[cfg1, cfg2],
                data_storage_path=str(tmp_path),
                show_progress_bars=False,
            )

    def test_duplicate_data_source_ids_raises(self, tmp_path):
        class DupIdSource(SimpleMockDataSource):
            def iter_ids(self):
                yield 1
                yield 1
                yield 2

        ds = DupIdSource(3)
        consumer = NodeConfig(
            MockNode(requires=["text"]),
            greedy=True,
            data_store=CSVDataStore,
        )

        with pytest.raises(AssertionError, match="Duplicate IDs"):
            BenchmarkManager(
                data_source=ds,
                nodes=[],
                consumer_nodes=[consumer],
                data_storage_path=str(tmp_path),
                show_progress_bars=False,
            )


class TestBenchmarkManagerStates:
    def test_pending_before_run(self, tmp_path):
        ds = SimpleMockDataSource(4)
        consumer = NodeConfig(
            MockNode(requires=["text"]),
            greedy=True,
            data_store=CSVDataStore,
        )
        mgr = BenchmarkManager(
            data_source=ds,
            nodes=[],
            consumer_nodes=[consumer],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
        )
        assert mgr.get_state() == "pending"

    def test_finished_after_run(self, tmp_path):
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
        )
        mgr.run_benchmark()
        assert mgr.get_state() == "finished"

    def test_state_with_no_runs(self):
        ds = SimpleMockDataSource(4)
        consumer = NodeConfig(
            MockNode(requires=["text"]),
            greedy=True,
            data_store=CSVDataStore,
        )
        mgr = BenchmarkManager(
            data_source=ds,
            nodes=[],
            consumer_nodes=[consumer],
            data_storage_path="/tmp/bench_state_test",
            show_progress_bars=False,
        )
        assert mgr.get_state() == "pending"

    def test_cannot_rerun(self, tmp_path):
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
            raise_exceptions=False,
        )
        mgr.run_benchmark()
        with pytest.raises(AssertionError):
            mgr.run_benchmark()


class TestBenchmarkManagerRunFails:
    def test_node_raises_sets_crashed(self, tmp_path):
        ds = SimpleMockDataSource(4)

        class ErrorNode:
            requires = ["text"]
            provides = ["processed"]

            def __call__(self, item_stats, **kwargs):
                raise ValueError("intentional error")

        error_cfg = NodeConfig(ErrorNode())
        consumer = NodeConfig(
            MockNode(requires=["processed"]),
            greedy=True,
            data_store=CSVDataStore,
        )
        mgr = BenchmarkManager(
            data_source=ds,
            nodes=[error_cfg],
            consumer_nodes=[consumer],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
            halt_on_exception=False,
            raise_exceptions=False,
        )
        mgr.run_benchmark()
        assert mgr.get_state() == "crashed"


class TestBenchmarkManagerNodeState:
    def test_active_node(self, tmp_path):
        ds = SimpleMockDataSource(4)
        node = MockNode(requires=["value"], provides=["out"])
        cfg = NodeConfig(node, id="processing_node")
        consumer = NodeConfig(
            MockNode(requires=["out"]),
            id="active_consumer",
            greedy=True,
            data_store=CSVDataStore,
        )
        mgr = BenchmarkManager(
            data_source=ds,
            nodes=[cfg, consumer],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
            raise_exceptions=False,
        )
        mgr.run_benchmark()
        state = mgr.get_node_state(cfg, mgr.base_adg)
        assert state == "active"

    def test_prued_node(self, tmp_path):
        ds = SimpleMockDataSource(4)
        unused = MockNode(requires=["value"], provides=["unused_out"])
        unused_cfg = NodeConfig(unused, id="unused_node")
        consumer = NodeConfig(
            MockNode(requires=["value"]),
            id="prune_consumer",
            greedy=True,
            data_store=CSVDataStore,
        )
        mgr = BenchmarkManager(
            data_source=ds,
            nodes=[unused_cfg, consumer],
            data_storage_path=str(tmp_path),
            show_progress_bars=False,
            raise_exceptions=False,
        )
        mgr.run_benchmark()
        state = mgr.get_node_state(unused_cfg, mgr.base_adg)
        assert state == "pruned"


class TestBenchmarkManagerReport:
    def test_report_json(self, tmp_path):
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
            raise_exceptions=False,
        )
        mgr.run_benchmark()
        report = mgr.get_report()
        j = report.to_json()
        assert "state" in j
        assert "nodes" in j
        assert j["state"] == "finished"

    def test_report_csv_row(self, tmp_path):
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
            raise_exceptions=False,
        )
        mgr.run_benchmark()
        report = mgr.get_report()
        header, row = report.to_csv_row()
        assert "State" in header
        assert "RunTime" in header


class TestBenchmarkManagerCSVOutput:
    def test_csv_written(self, tmp_path):
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
            raise_exceptions=False,
        )
        mgr.run_benchmark()
        report = mgr.get_report()
        csv_path = str(tmp_path / "report.csv")
        report.write_csv_to_file(csv_path)
        import os

        assert os.path.exists(csv_path)
