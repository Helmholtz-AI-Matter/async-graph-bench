import pytest
from async_graph_bench.stores.csv_store import safe_literal_eval, CSVDataStore


class TestSafeLiteralEval:
    def test_int(self):
        assert safe_literal_eval("42") == 42

    def test_float(self):
        assert safe_literal_eval("3.14") == 3.14

    def test_bool_true(self):
        assert safe_literal_eval("True") is True

    def test_bool_false(self):
        assert safe_literal_eval("False") is False

    def test_none(self):
        assert safe_literal_eval("None") is None

    def test_string_fallback(self):
        assert safe_literal_eval("hello world") == "hello world"

    def test_list(self):
        assert safe_literal_eval("[1, 2, 3]") == [1, 2, 3]

    def test_dict(self):
        assert safe_literal_eval("{'a': 1}") == {"a": 1}

    def test_empty_string(self):
        assert safe_literal_eval("") == ""


class TestCSVDataStore:
    def test_create_store(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        assert store.filepath == tmp_path / "test.csv"

    def test_save_load_single(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "value": "hello", "iter": 0})
        store.flush()
        result = store.load(1, 0)
        assert result["value"] == "hello"

    def test_save_load_multiple(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        store.save({"id": 2, "val": 20})
        store.flush()
        assert store.load(1, 0)["val"] == 10
        assert store.load(2, 0)["val"] == 20

    def test_delete(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        assert store.delete(1, 0) is True
        assert store.delete(1, 0) is False

    def test_contains_key(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        assert store.contains_key(1, 0) is True
        assert store.contains_key(2, 0) is False

    def test_load_missing_returns_none(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        assert store.load(999, 0) is None

    def test_file_not_found_no_create_okay(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CSVDataStore(str(tmp_path), "nonexistent", create_okay=False)

    def test_clear(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        store.flush()
        store.clear()
        assert len(store) == 0
        assert not store.filepath.exists()

    def test_len(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        assert len(store) == 0
        store.save({"id": 1, "val": 10})
        assert len(store) == 1

    def test_iter_keys(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "iter": 0, "val": 10})
        store.save({"id": 2, "iter": 1, "val": 20})
        keys = list(store.iter_keys())
        assert (1, 0) in keys
        assert (2, 1) in keys

    def test_save_update_existing(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        store.save({"id": 1, "val": 99})
        assert len(store) == 1
        assert store.load(1, 0)["val"] == 99

    def test_to_dataframe(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10, "name": "a"})
        store.save({"id": 2, "val": 20, "name": "b"})
        df = store.to_dataframe()
        assert len(df) == 2
        assert "val" in df.columns

    def test_to_dataframe_with_properties(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10, "name": "a"})
        df = store.to_dataframe(properties=["id", "val"])
        assert "name" not in df.columns

    def test_column_ordering(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "iter": 0, "zebra": 1, "alpha": 2, "mango": 3})
        store.flush()
        with open(store.filepath) as f:
            header = f.readline().strip()
        parts = header.split(";")
        zebra_idx = parts.index("zebra")
        alpha_idx = parts.index("alpha")
        mango_idx = parts.index("mango")
        assert alpha_idx < mango_idx < zebra_idx

    def test_persist_to_disk(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 42})
        store.flush()
        store2 = CSVDataStore(str(tmp_path), "test", create_okay=True)
        assert store2.load(1, 0)["val"] == 42

    def test_separator_is_semicolon(self, tmp_path):
        store = CSVDataStore(str(tmp_path), "test", create_okay=True)
        assert store.separator == ";"