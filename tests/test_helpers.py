import pytest
from bitarray import bitarray
from async_graph_bench.utils.helpers import (
    get_duplicates,
    build_combined_keys,
    is_fully_resolved,
    expand_resolved_ids,
    adjust_string_length,
    flatten_recursive,
    check_unique_strings,
    resolved_ids_to_bitarray,
    get_metadata,
    update_metadata,
    clear_metadata,
)


class TestGetDuplicates:
    def test_no_duplicates(self):
        assert get_duplicates([1, 2, 3, 4]) == set()

    def test_single_duplicate(self):
        assert get_duplicates([1, 2, 2, 3]) == {2}

    def test_multiple_duplicates(self):
        assert get_duplicates([1, 1, 2, 2, 3]) == {1, 2}

    def test_all_same(self):
        assert get_duplicates([5, 5, 5]) == {5}

    def test_empty(self):
        assert get_duplicates([]) == set()


class TestCheckUniqueStrings:
    def test_unique(self):
        check_unique_strings(["a", "b", "c"])

    def test_duplicate_raises(self):
        with pytest.raises(Exception, match="Duplicate string found"):
            check_unique_strings(["a", "b", "b"])

    def test_empty(self):
        check_unique_strings([])

    def test_single(self):
        check_unique_strings(["only"])


class TestBuildCombinedKeys:
    def test_iterations_first(self):
        keys = build_combined_keys([0, 1], iterations=3, iterations_first=True)
        assert keys == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]

    def test_not_iterations_first(self):
        keys = build_combined_keys([0, 1], iterations=3, iterations_first=False)
        assert keys == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    def test_single_iteration_first(self):
        keys = build_combined_keys([0, 1], iterations=1, iterations_first=True)
        assert keys == [(0, 0), (0, 1)]

    def test_tuple_ids_iterations_first(self):
        keys = build_combined_keys([(1, 2), (3, 4)], iterations=2, iterations_first=True)
        assert keys == [(0, 1, 2), (1, 1, 2), (0, 3, 4), (1, 3, 4)]

    def test_tuple_ids_not_iterations_first(self):
        keys = build_combined_keys([(1, 2), (3, 4)], iterations=2, iterations_first=False)
        assert keys == [(0, 1, 2), (0, 3, 4), (1, 1, 2), (1, 3, 4)]

    def test_zero_iterations(self):
        keys = build_combined_keys([0, 1], iterations=0, iterations_first=True)
        assert keys == []

    def test_string_ids(self):
        keys = build_combined_keys(["a", "b"], iterations=2, iterations_first=True)
        assert keys == [(0, "a"), (1, "a"), (0, "b"), (1, "b")]


class TestExpandResolvedIds:
    def test_basic_expansion(self):
        resolved = {(0, 1)}
        result = expand_resolved_ids(resolved, iterations=4, sample_size=4)
        assert result == {(0, 1), (1, 1), (2, 1), (3, 1)}

    def test_multiple_ids(self):
        resolved = {(0, 1), (0, 2)}
        result = expand_resolved_ids(resolved, iterations=8, sample_size=4)
        assert (0, 1) in result
        assert (3, 1) in result
        assert (0, 2) in result
        assert (3, 2) in result

    def test_sample_size_one(self):
        resolved = {(0, 1), (1, 2)}
        result = expand_resolved_ids(resolved, iterations=2, sample_size=1)
        assert result == {(0, 1), (1, 2)}

    def test_empty(self):
        result = expand_resolved_ids(set(), iterations=4, sample_size=2)
        assert result == set()


class TestResolvedIdsToBitarray:
    def test_all_resolved(self):
        idx_map = {(0, 1): 0, (1, 1): 1, (0, 2): 2}
        resolved = {(0, 1), (1, 1), (0, 2)}
        ba = resolved_ids_to_bitarray(idx_map, resolved)
        assert ba.count(1) == 3

    def test_none_resolved(self):
        idx_map = {(0, 1): 0, (1, 1): 1}
        ba = resolved_ids_to_bitarray(idx_map, set())
        assert ba.count(1) == 0

    def test_partial(self):
        idx_map = {(0, 1): 0, (1, 1): 1, (0, 2): 2}
        ba = resolved_ids_to_bitarray(idx_map, [(0, 1)])
        assert ba.count(1) == 1

    def test_unknown_id_ignored(self):
        idx_map = {(0, 1): 0}
        ba = resolved_ids_to_bitarray(idx_map, [(0, 1), (99, 99)])
        assert ba.count(1) == 1


class TestIsFullyResolved:
    def test_exact_match(self):
        idx_map = {(0, 1): 0, (1, 1): 1}
        resolved = {(0, 1), (1, 1)}
        assert is_fully_resolved(idx_map, resolved) is True

    def test_one_missing(self):
        idx_map = {(0, 1): 0, (1, 1): 1}
        resolved = {(0, 1)}
        assert is_fully_resolved(idx_map, resolved) is False

    def test_extra_ids(self):
        idx_map = {(0, 1): 0}
        resolved = {(0, 1), (99, 99)}
        assert is_fully_resolved(idx_map, resolved) is True

    def test_empty_both(self):
        assert is_fully_resolved({}, set()) is True


class TestAdjustStringLength:
    def test_truncate(self):
        assert adjust_string_length("hello world", 5) == "hello"

    def test_pad(self):
        assert adjust_string_length("hi", 5) == "hi   "

    def test_exact(self):
        assert adjust_string_length("abc", 3) == "abc"

    def test_empty(self):
        assert adjust_string_length("", 3) == "   "


class TestFlattenRecursive:
    def test_already_flat(self):
        assert list(flatten_recursive([1, 2, 3])) == [1, 2, 3]

    def test_nested_lists(self):
        assert list(flatten_recursive([1, [2, 3], [4, [5, 6]]])) == [1, 2, 3, 4, 5, 6]

    def test_nested_tuples(self):
        assert list(flatten_recursive((1, (2, 3)))) == [1, 2, 3]

    def test_deeply_nested(self):
        assert list(flatten_recursive([[[1]]])) == [1]

    def test_empty(self):
        assert list(flatten_recursive([])) == []

    def test_mixed(self):
        assert list(flatten_recursive([1, [2, (3, [4])]])) == [1, 2, 3, 4]

    def test_non_iterable_items(self):
        assert list(flatten_recursive([1, 2, "three"])) == [1, 2, "three"]


class TestMetadata:
    def test_get_metadata_missing(self, tmp_path):
        assert get_metadata(str(tmp_path), "test") == {}

    def test_update_and_get_metadata(self, tmp_path):
        update_metadata(str(tmp_path), "test", {"key": "value"})
        result = get_metadata(str(tmp_path), "test")
        assert result["key"] == "value"

    def test_update_merges_metadata(self, tmp_path):
        update_metadata(str(tmp_path), "test", {"a": 1})
        update_metadata(str(tmp_path), "test", {"b": 2})
        result = get_metadata(str(tmp_path), "test")
        assert result == {"a": 1, "b": 2}

    def test_update_metadata_not_dict_raises(self, tmp_path):
        with pytest.raises(TypeError):
            update_metadata(str(tmp_path), "test", "not a dict")

    def test_clear_metadata(self, tmp_path):
        update_metadata(str(tmp_path), "test", {"a": 1})
        clear_metadata(str(tmp_path), "test")
        assert get_metadata(str(tmp_path), "test") == {}

    def test_clear_missing_metadata_noop(self, tmp_path):
        clear_metadata(str(tmp_path), "test")