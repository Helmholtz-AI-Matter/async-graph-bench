import pytest
from async_graph_bench.stores.diskcache_store import (
    DiskCacheStore,
    truncate_innermost_arrays,
)
import numpy as np


class TestDiskCacheStore:
    def test_create_store(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        assert store.cache_path == str(tmp_path / "test")

    def test_save_load_single(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "value": "hello", "iter": 0})
        result = store.load(1, 0)
        assert result["value"] == "hello"

    def test_save_load_multiple(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        store.save({"id": 2, "val": 20})
        assert store.load(1, 0)["val"] == 10
        assert store.load(2, 0)["val"] == 20

    def test_save_with_iter(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "iter": 3, "val": 42})
        assert store.load(1, 3)["val"] == 42

    def test_delete(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        store.delete(1, 0)
        assert store.load(1, 0) is None

    def test_contains_key(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        assert store.contains_key(1, 0) is True
        assert store.contains_key(2, 0) is False

    def test_load_missing_returns_none(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        assert store.load(999, 0) is None

    def test_file_not_found_no_create_okay(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DiskCacheStore(str(tmp_path), "nonexistent", create_okay=False)

    def test_clear(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        store.clear()
        assert len(store) == 0

    def test_len(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        assert len(store) == 0
        store.save({"id": 1, "val": 10})
        assert len(store) == 1
        store.save({"id": 2, "val": 20})
        assert len(store) == 2

    def test_iter_keys(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "iter": 0, "val": 10})
        store.save({"id": 2, "iter": 1, "val": 20})
        keys = list(store.iter_keys())
        assert (1, 0) in keys
        assert (2, 1) in keys

    def test_iter_items(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 42})
        items = list(store.iter_items())
        assert len(items) == 1
        assert items[0]["val"] == 42

    def test_flush_is_noop(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        store.flush()

    def test_to_dataframe(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10, "name": "a"})
        store.save({"id": 2, "val": 20, "name": "b"})
        df = store.to_dataframe()
        assert len(df) == 2
        assert "val" in df.columns

    def test_to_dataframe_with_properties(self, tmp_path):
        store = DiskCacheStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10, "name": "a"})
        df = store.to_dataframe(properties=["id", "val"])
        assert "name" not in df.columns


class TestTruncateInnermostArrays:
    def test_uniform_arrays_unchanged(self):
        arr = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=object
        )
        result = truncate_innermost_arrays(arr)
        assert np.array_equal(result[0][0], np.array([1, 2, 3]))

    def test_truncates_to_min_length(self):
        arr = np.array([[[1, 2, 3, 4], [5, 6]], [[7, 8], [9, 10, 11]]], dtype=object)
        result = truncate_innermost_arrays(arr)
        assert len(result[0][0]) == 2
        assert len(result[0][1]) == 2
        assert np.array_equal(result[0][0], np.array([1, 2]))
