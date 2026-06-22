import pytest
from tests.fixtures import SimpleMockDataSource, AsyncMockDataSource


@pytest.fixture
def tmp_storage_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def mock_data_source():
    return SimpleMockDataSource(num_items=4)


@pytest.fixture
def async_mock_data_source():
    return AsyncMockDataSource(num_items=4)
