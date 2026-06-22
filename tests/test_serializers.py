from async_graph_bench.stores.serializers.pickle import PickleSerializer
from async_graph_bench.stores.serializers.zlib import ZLibCompressionSerializer
from async_graph_bench.stores.serializers.message_pack import MessagePackSerializer
from async_graph_bench.stores.serializers.zstandard import ZstdCompressionSerializer


class TestPickleSerializer:
    def test_roundtrip_dict(self):
        s = PickleSerializer()
        original = {"a": 1, "b": [2, 3, 4], "c": {"nested": True}}
        assert s.deserialize(s.serialize(original)) == original

    def test_roundtrip_list(self):
        s = PickleSerializer()
        original = [1, "two", 3.0, None, True]
        assert s.deserialize(s.serialize(original)) == original

    def test_roundtrip_scalar(self):
        s = PickleSerializer()
        for val in [42, 3.14, "hello", True, False, None]:
            assert s.deserialize(s.serialize(val)) == val

    def test_roundtrip_nested(self):
        s = PickleSerializer()
        original = {"a": [{"b": {"c": [1, 2, 3]}}]}
        assert s.deserialize(s.serialize(original)) == original


class TestZLibCompressionSerializer:
    def test_roundtrip_bytes(self):
        s = ZLibCompressionSerializer()
        original = b"hello world this is a test of zlib compression"
        assert s.deserialize(s.serialize(original)) == original

    def test_roundtrip_empty(self):
        s = ZLibCompressionSerializer()
        assert s.deserialize(s.serialize(b"")) == b""

    def test_custom_level(self):
        s = ZLibCompressionSerializer(level=9)
        original = b"compressible data " * 100
        compressed = s.serialize(original)
        assert len(compressed) < len(original)
        assert s.deserialize(compressed) == original

    def test_returns_bytes(self):
        s = ZLibCompressionSerializer()
        assert isinstance(s.serialize(b"test"), bytes)


class TestZstdCompressionSerializer:
    def test_roundtrip(self):
        s = ZstdCompressionSerializer()
        original = b"zstandard compression test data" * 50
        compressed = s.serialize(original)
        assert s.deserialize(compressed) == original

    def test_custom_level(self):
        s = ZstdCompressionSerializer(level=1)
        original = b"test data" * 100
        assert s.deserialize(s.serialize(original)) == original


class TestMessagePackSerializer:
    def test_roundtrip_dict(self):
        s = MessagePackSerializer()
        original = {"a": 1, "b": "two", "c": [3, 4, 5]}
        assert s.deserialize(s.serialize(original)) == original

    def test_roundtrip_list(self):
        s = MessagePackSerializer()
        original = [1, "two", 3.0, True, False, None]
        assert s.deserialize(s.serialize(original)) == original

    def test_roundtrip_nested(self):
        s = MessagePackSerializer()
        original = {"outer": {"inner": [1, 2, {"deep": "value"}]}}
        assert s.deserialize(s.serialize(original)) == original


class TestZLibCompressionRatio:
    def test_zlib_compresses_repetitive_data(self):
        s = ZLibCompressionSerializer()
        original = b"repetitive data for compression test " * 100
        compressed = s.serialize(original)
        assert len(compressed) < len(original)
        assert s.deserialize(compressed) == original

    def test_zlib_compression_level_9_smallest(self):
        s_low = ZLibCompressionSerializer(level=1)
        s_high = ZLibCompressionSerializer(level=9)
        original = b"compress me please " * 200
        low = s_low.serialize(original)
        high = s_high.serialize(original)
        assert len(high) <= len(low)
        assert s_low.deserialize(low) == original
        assert s_high.deserialize(high) == original

    def test_zlib_no_compression_level_0(self):
        s = ZLibCompressionSerializer(level=0)
        original = b"no compression " * 100
        compressed = s.serialize(original)
        assert s.deserialize(compressed) == original


class TestZstdCompressionRatio:
    def test_zstd_compresses_repetitive_data(self):
        s = ZstdCompressionSerializer()
        original = b"repetitive data for zstd compression test " * 100
        compressed = s.serialize(original)
        assert len(compressed) < len(original)
        assert s.deserialize(compressed) == original

    def test_zstd_higher_level_better_compression(self):
        s_low = ZstdCompressionSerializer(level=1)
        s_high = ZstdCompressionSerializer(level=10)
        original = b"compress me with zstd " * 200
        low = s_low.serialize(original)
        high = s_high.serialize(original)
        assert len(high) <= len(low)
        assert s_low.deserialize(low) == original
        assert s_high.deserialize(high) == original

    def test_zstd_large_data(self):
        s = ZstdCompressionSerializer()
        original = b"a" * 10000
        compressed = s.serialize(original)
        assert len(compressed) < len(original)
        assert s.deserialize(compressed) == original


class TestSerializerChains:
    def test_pickle_then_zlib(self):
        pickle_ser = PickleSerializer()
        zlib_ser = ZLibCompressionSerializer()
        original = {"key": "value", "numbers": [1, 2, 3]}
        serialized = zlib_ser.serialize(pickle_ser.serialize(original))
        restored = pickle_ser.deserialize(zlib_ser.deserialize(serialized))
        assert restored == original

    def test_pickle_then_zstd(self):
        pickle_ser = PickleSerializer()
        zstd_ser = ZstdCompressionSerializer()
        original = {"key": "value", "nested": {"a": [1, 2, 3]}}
        serialized = zstd_ser.serialize(pickle_ser.serialize(original))
        restored = pickle_ser.deserialize(zstd_ser.deserialize(serialized))
        assert restored == original
