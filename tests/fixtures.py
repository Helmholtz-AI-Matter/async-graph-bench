from typing import List, Dict, Any, Iterator
from async_graph_bench import DataSource


class SimpleMockDataSource(DataSource):
    """A simple synchronous DataSource for testing."""

    def __init__(self, num_items: int = 5):
        self._num_items = num_items

    @property
    def provides(self) -> List[str]:
        return ["text", "value"]

    def __len__(self) -> int:
        return self._num_items

    def iter_items(self) -> Iterator[Dict[str, Any]]:
        for i in range(self._num_items):
            yield {"id": i, "text": f"item_{i}", "value": float(i)}

    def iter_ids(self) -> Iterator[int]:
        for i in range(self._num_items):
            yield i


class AsyncMockDataSource(DataSource):
    """An asynchronous DataSource for testing."""

    def __init__(self, num_items: int = 5):
        self._num_items = num_items

    @property
    def provides(self) -> List[str]:
        return ["text", "value"]

    def __len__(self) -> int:
        return self._num_items

    async def iter_items(self):
        for i in range(self._num_items):
            yield {"id": i, "text": f"item_{i}", "value": float(i)}

    def iter_ids(self) -> Iterator[int]:
        for i in range(self._num_items):
            yield i


class MockNode:
    """A node implementing the Node protocol for testing."""

    def __init__(self, requires: List[str], provides: List[str] | None = None,
                 async_call: bool = False, multiply: float = 1.0):
        self.requires = requires
        if provides is not None:
            self.provides = provides
        self._async = async_call
        self._multiply = multiply
        self.call_count = 0

    def __call__(self, item_stats: Dict[str, list], **kwargs) -> Dict[str, list] | Any:
        self.call_count += 1
        if hasattr(self, 'provides'):
            first_input = item_stats[self.requires[0]]
            return {self.provides[0]: [v * self._multiply for v in first_input]}
        first_input = item_stats[self.requires[0]]
        return [v * self._multiply for v in first_input]
