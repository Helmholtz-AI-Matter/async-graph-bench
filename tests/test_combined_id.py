from async_graph_bench.stores.combined_id import get_combined_id, get_combined_id_from_parts


class TestGetCombinedId:
    def test_scalar_id_default_iter(self):
        item = {"id": 5}
        assert get_combined_id(item) == (0, 5)

    def test_scalar_id_with_iter(self):
        item = {"id": 5, "iter": 3}
        assert get_combined_id(item) == (3, 5)

    def test_tuple_id_default_iter(self):
        item = {"id": (1, 2)}
        assert get_combined_id(item) == (0, 1, 2)

    def test_tuple_id_with_iter(self):
        item = {"id": (1, 2), "iter": 7}
        assert get_combined_id(item) == (7, 1, 2)

    def test_string_id(self):
        item = {"id": "abc"}
        assert get_combined_id(item) == (0, "abc")

    def test_string_id_with_iter(self):
        item = {"id": "abc", "iter": 2}
        assert get_combined_id(item) == (2, "abc")


class TestGetCombinedIdFromParts:
    def test_scalar_id_default_iter(self):
        assert get_combined_id_from_parts(5) == (0, 5)

    def test_scalar_id_with_iter(self):
        assert get_combined_id_from_parts(5, 3) == (3, 5)

    def test_tuple_id_default_iter(self):
        assert get_combined_id_from_parts((1, 2)) == (0, 1, 2)

    def test_tuple_id_with_iter(self):
        assert get_combined_id_from_parts((1, 2), 7) == (7, 1, 2)

    def test_string_id(self):
        assert get_combined_id_from_parts("abc") == (0, "abc")

    def test_consistency_with_get_combined_id(self):
        for item in [
            {"id": 5}, {"id": 5, "iter": 3},
            {"id": (1, 2)}, {"id": (1, 2), "iter": 7},
            {"id": "x"}, {"id": "x", "iter": 2},
        ]:
            result1 = get_combined_id(item)
            result2 = get_combined_id_from_parts(item["id"], item.get("iter", 0))
            assert result1 == result2