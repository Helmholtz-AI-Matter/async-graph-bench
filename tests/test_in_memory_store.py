import pytest
import pandas as pd
from async_graph_bench.stores.json_store import JSONDataStore


class TestInMemoryStore:
    def test_save_append_new(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        assert len(store) == 1

    def test_save_update_existing(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        store.save({"id": 1, "val": 99})
        assert len(store) == 1
        assert store.load(1, 0)["val"] == 99

    def test_save_update_existing_with_iter(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "iter": 2, "val": 10})
        store.save({"id": 1, "iter": 2, "val": 99})
        assert len(store) == 1
        assert store.load(1, 2)["val"] == 99

    def test_save_different_iter_appends(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "iter": 0, "val": 10})
        store.save({"id": 1, "iter": 1, "val": 20})
        assert len(store) == 2

    def test_delete_existing(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        assert store.delete(1, 0) is True
        assert len(store) == 0

    def test_delete_nonexistent(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        assert store.delete(999, 0) is False

    def test_delete_with_iter(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "iter": 3, "val": 10})
        assert store.delete(1, 3) is True
        assert store.delete(1, 0) is False

    def test_contains_key_true(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        assert store.contains_key(1, 0) is True

    def test_contains_key_false(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        assert store.contains_key(2, 0) is False

    def test_contains_key_with_iter(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "iter": 5, "val": 10})
        assert store.contains_key(1, 5) is True
        assert store.contains_key(1, 0) is False

    def test_load_existing(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 42, "name": "test"})
        result = store.load(1, 0)
        assert result["val"] == 42
        assert result["name"] == "test"

    def test_load_nonexistent(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        assert store.load(999, 0) is None

    def test_load_default_iter_matches_no_iter(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        assert store.load(1, 0) is not None

    def test_iter_keys(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "iter": 0})
        store.save({"id": 2, "iter": 1})
        keys = list(store.iter_keys())
        assert (1, 0) in keys
        assert (2, 1) in keys

    def test_iter_items(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        items = list(store.iter_items())
        assert len(items) == 1
        assert items[0]["val"] == 10

    def test_len(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        assert len(store) == 0
        store.save({"id": 1, "val": 10})
        assert len(store) == 1
        store.save({"id": 2, "val": 20})
        assert len(store) == 2

    def test_to_dataframe_all(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10, "name": "a"})
        store.save({"id": 2, "val": 20, "name": "b"})
        df = store.to_dataframe()
        assert len(df) == 2
        assert set(df.columns) == {"id", "val", "name"}

    def test_to_dataframe_subset(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10, "name": "a"})
        df = store.to_dataframe(properties=["id", "val"])
        assert "name" not in df.columns

    def test_to_dataframe_empty(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        df = store.to_dataframe()
        assert len(df) == 0

    def test_flush_every_auto_flush(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True, flush_every=2)
        store.save({"id": 1, "val": 10})
        store.save({"id": 2, "val": 20})
        store2 = JSONDataStore(str(tmp_path), "test", create_okay=True)
        assert store2.load(1, 0)["val"] == 10
        assert store2.load(2, 0)["val"] == 20

    def test_clear(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        store.flush()
        store.clear()
        assert len(store) == 0
        assert not store.filepath.exists()