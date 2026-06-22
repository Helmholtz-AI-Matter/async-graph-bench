from async_graph_bench import (
    BenchmarkManager,
    NodeConfig,
    CSVDataStore,
)
from tests.fixtures import MockNode, SimpleMockDataSource


class TestBenchmarkReportJSON:
    def test_json_structure(self, tmp_path):
        ds = SimpleMockDataSource(3)
        consumer = NodeConfig(
            MockNode(requires=["value"]),
            greedy=True,
            data_store=CSVDataStore,
            id="my_consumer",
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

        assert j["state"] == "finished"
        assert j["total_steps"] == 1
        assert "total_time" in j
        assert "my_consumer" in j["nodes"]


class TestBenchmarkReportCSVRow:
    def test_csv_row_columns(self, tmp_path):
        ds = SimpleMockDataSource(3)
        consumer = NodeConfig(
            MockNode(requires=["value"]),
            greedy=True,
            data_store=CSVDataStore,
            id="c1",
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

        assert "c1" in header
        assert "State" in header
        assert "RunTime" in header
        assert "FinishTime" in header

        state_idx = header.index("State")
        assert row[state_idx] == "finished"


class TestBenchmarkReportTable:
    def test_to_table_output(self, tmp_path):
        ds = SimpleMockDataSource(3)
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
        table = report.to_table()
        assert isinstance(table, str)
        assert "\u250c" in table or "\u2500" in table or "Node" in table

    def test_to_markdown_table(self, tmp_path):
        ds = SimpleMockDataSource(3)
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
        md = report.to_markdown_table()
        assert isinstance(md, str)
        assert "|" in md


class TestBenchmarkReportCSVFile:
    def test_write_csv_fresh(self, tmp_path):
        ds = SimpleMockDataSource(3)
        consumer = NodeConfig(
            MockNode(requires=["value"]),
            greedy=True,
            data_store=CSVDataStore,
            id="x",
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
        csv_path = str(tmp_path / "results.csv")
        report.write_csv_to_file(csv_path)
        import csv

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert "x" in rows[0]

    def test_write_csv_append_matching(self, tmp_path):
        ds = SimpleMockDataSource(3)
        consumer = NodeConfig(
            MockNode(requires=["value"]),
            greedy=True,
            data_store=CSVDataStore,
            id="y",
        )
        csv_path = str(tmp_path / "results.csv")

        for i in range(2):
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
            report.write_csv_to_file(csv_path)

        import csv

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 3
