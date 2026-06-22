import pytest
from async_graph_bench.stores.json_store import encode_tuples, decode_tuples, JSONDataStore
import os


class TestEncodeDecodeTuples:
    def test_encode_simple_tuple(self):
        assert encode_tuples((1, 2, 3)) == {"__tuple__": True, "items": [1, 2, 3]}

    def test_decode_simple_tuple(self):
        assert decode_tuples({"__tuple__": True, "items": [1, 2, 3]}) == (1, 2, 3)

    def test_roundtrip_simple_tuple(self):
        original = (1, "two", 3.0)
        assert decode_tuples(encode_tuples(original)) == original

    def test_roundtrip_nested_tuples(self):
        original = {"key": ((1, 2), [3, (4, 5)])}
        assert decode_tuples(encode_tuples(original)) == original

    def test_roundtrip_no_tuples(self):
        original = {"a": 1, "b": [2, 3, 4]}
        assert decode_tuples(encode_tuples(original)) == original

    def test_roundtrip_deeply_nested(self):
        original = (((1,), (2,)), {"a": (3, 4)})
        assert decode_tuples(encode_tuples(original)) == original

    def test_empty_tuple(self):
        original = ()
        assert decode_tuples(encode_tuples(original)) == original

    def test_tuple_in_list(self):
        original = [(1, 2), (3, 4), "five"]
        assert decode_tuples(encode_tuples(original)) == original


class TestJSONDataStore:
    def test_create_store(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        assert store.filepath == tmp_path / "test.json"

    def test_save_load_single(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "value": "hello", "iter": 0})
        store.flush()
        result = store.load(1, 0)
        assert result["value"] == "hello"

    def test_save_load_multiple(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        store.save({"id": 2, "val": 20})
        store.flush()
        assert store.load(1, 0)["val"] == 10
        assert store.load(2, 0)["val"] == 20

    def test_delete(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        assert store.delete(1, 0) is True
        assert store.delete(1, 0) is False

    def test_contains_key(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        assert store.contains_key(1, 0) is True
        assert store.contains_key(2, 0) is False

    def test_load_missing_returns_none(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        assert store.load(999, 0) is None

    def test_file_not_found_no_create_okay(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            JSONDataStore(str(tmp_path), "nonexistent", create_okay=False)

    def test_clear(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        store.flush()
        store.clear()
        assert len(store) == 0
        assert not store.filepath.exists()

    def test_len(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        assert len(store) == 0
        store.save({"id": 1, "val": 10})
        assert len(store) == 1
        store.save({"id": 2, "val": 20})
        assert len(store) == 2

    def test_iter_keys(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "iter": 0, "val": 10})
        store.save({"id": 2, "iter": 1, "val": 20})
        keys = list(store.iter_keys())
        assert (1, 0) in keys
        assert (2, 1) in keys

    def test_save_update_existing(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10})
        store.save({"id": 1, "val": 99})
        assert len(store) == 1
        assert store.load(1, 0)["val"] == 99

    def test_to_dataframe(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10, "name": "a"})
        store.save({"id": 2, "val": 20, "name": "b"})
        df = store.to_dataframe()
        assert len(df) == 2
        assert "val" in df.columns

    def test_to_dataframe_with_properties(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 10, "name": "a"})
        df = store.to_dataframe(properties=["id", "val"])
        assert "name" not in df.columns

    def test_persist_to_disk(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "val": 42})
        store.flush()
        store2 = JSONDataStore(str(tmp_path), "test", create_okay=True)
        assert store2.load(1, 0)["val"] == 42

    def test_tuple_preservation_through_store(self, tmp_path):
        store = JSONDataStore(str(tmp_path), "test", create_okay=True)
        store.save({"id": 1, "data": (1, 2, 3)})
        store.flush()
        store2 = JSONDataStore(str(tmp_path), "test", create_okay=True)
        result = store2.load(1, 0)
        assert result["data"] == (1, 2, 3)
