import pytest
from async_graph_bench import DataSourcePartitioner
from tests.fixtures import SimpleMockDataSource, AsyncMockDataSource


class TestDataSource:
    def test_simple_mock_source_length(self):
        ds = SimpleMockDataSource(5)
        assert len(ds) == 5

    def test_simple_mock_source_provides(self):
        ds = SimpleMockDataSource()
        assert "text" in ds.provides
        assert "value" in ds.provides

    def test_simple_mock_source_id(self):
        ds = SimpleMockDataSource()
        assert ds.id == "SimpleMockDataSource"

    def test_simple_mock_source_iter_items(self):
        ds = SimpleMockDataSource(3)
        items = list(ds.iter_items())
        assert len(items) == 3
        assert items[0]["id"] == 0
        assert items[0]["text"] == "item_0"

    def test_simple_mock_source_iter_ids(self):
        ds = SimpleMockDataSource(4)
        ids = list(ds.iter_ids())
        assert ids == [0, 1, 2, 3]

    def test_async_mock_source_length(self):
        ds = AsyncMockDataSource(7)
        assert len(ds) == 7

    def test_async_mock_source_iter_ids(self):
        ds = AsyncMockDataSource(3)
        ids = list(ds.iter_ids())
        assert ids == [0, 1, 2]


class TestDataSourcePartitioner:
    def test_basic_partitioning(self):
        source = SimpleMockDataSource(10)
        p = DataSourcePartitioner(source, num_splits=2, split_index=0)
        assert len(p) == 5
        ids = list(p.iter_ids())
        assert ids == [0, 1, 2, 3, 4]

    def test_second_partition(self):
        source = SimpleMockDataSource(10)
        p = DataSourcePartitioner(source, num_splits=2, split_index=1)
        ids = list(p.iter_ids())
        assert ids == [5, 6, 7, 8, 9]

    def test_remainder_distribution(self):
        source = SimpleMockDataSource(11)
        p0 = DataSourcePartitioner(source, num_splits=2, split_index=0)
        p1 = DataSourcePartitioner(source, num_splits=2, split_index=1)
        assert len(p0) == 6
        assert len(p1) == 5

    def test_three_way_split_remainder(self):
        source = SimpleMockDataSource(10)
        p0 = DataSourcePartitioner(source, num_splits=3, split_index=0)
        p1 = DataSourcePartitioner(source, num_splits=3, split_index=1)
        p2 = DataSourcePartitioner(source, num_splits=3, split_index=2)
        assert len(p0) == 4
        assert len(p1) == 3
        assert len(p2) == 3

    def test_single_split(self):
        source = SimpleMockDataSource(5)
        p = DataSourcePartitioner(source, num_splits=1, split_index=0)
        assert len(p) == 5
        assert list(p.iter_ids()) == [0, 1, 2, 3, 4]

    def test_empty_last_partition(self):
        source = SimpleMockDataSource(3)
        for i in range(3):
            DataSourcePartitioner(source, num_splits=3, split_index=i)
        ds = SimpleMockDataSource(1)
        p0 = DataSourcePartitioner(ds, num_splits=3, split_index=0)
        p1 = DataSourcePartitioner(ds, num_splits=3, split_index=1)
        assert len(p0) == 1
        assert len(p1) == 0

    def test_provides_delegates(self):
        source = SimpleMockDataSource(5)
        p = DataSourcePartitioner(source, num_splits=2, split_index=0)
        assert p.provides == source.provides

    def test_num_splits_zero_raises(self):
        source = SimpleMockDataSource(5)
        with pytest.raises(AssertionError):
            DataSourcePartitioner(source, num_splits=0, split_index=0)

    def test_split_index_out_of_range_raises(self):
        source = SimpleMockDataSource(5)
        with pytest.raises(AssertionError):
            DataSourcePartitioner(source, num_splits=2, split_index=2)

    def test_iter_items_sync(self):
        source = SimpleMockDataSource(10)
        p = DataSourcePartitioner(source, num_splits=2, split_index=1)
        items = list(p.iter_items())
        assert len(items) == 5
        assert items[0]["id"] == 5

    def test_async_source_partitioning(self):
        source = AsyncMockDataSource(10)
        p = DataSourcePartitioner(source, num_splits=2, split_index=0)
        assert len(p) == 5
        ids = list(p.iter_ids())
        assert ids == [0, 1, 2, 3, 4]

    def test_all_partitions_cover_all(self):
        source = SimpleMockDataSource(7)
        num_splits = 3
        all_ids = set()
        for i in range(num_splits):
            p = DataSourcePartitioner(source, num_splits=num_splits, split_index=i)
            all_ids.update(p.iter_ids())
        assert all_ids == {0, 1, 2, 3, 4, 5, 6}
