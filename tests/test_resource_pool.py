import pytest
import asyncio
from async_graph_bench.utils.resource_pool import (
    ResourcePool,
    acquire_from_many,
)


class TestResourcePool:
    def test_create(self):
        pool = ResourcePool([1, 2, 3])
        assert pool.total() == 3

    def test_available_equals_total_initial(self):
        pool = ResourcePool([1, 2, 3])
        assert pool.available() == 3

    @pytest.mark.asyncio
    async def test_acquire_single(self):
        pool = ResourcePool(["a", "b"])
        handle = await pool.acquire(1)
        assert pool.available() == 1
        await handle.release()
        assert pool.available() == 2

    @pytest.mark.asyncio
    async def test_acquire_multiple(self):
        pool = ResourcePool(["a", "b", "c"])
        handle = await pool.acquire(2)
        assert pool.available() == 1
        await handle.release()
        assert pool.available() == 3

    @pytest.mark.asyncio
    async def test_acquire_zero_raises(self):
        pool = ResourcePool([1, 2])
        with pytest.raises(ValueError):
            await pool.acquire(0)

    @pytest.mark.asyncio
    async def test_acquire_negative_raises(self):
        pool = ResourcePool([1, 2])
        with pytest.raises(ValueError):
            await pool.acquire(-1)

    @pytest.mark.asyncio
    async def test_resource_handle_aenter_single(self):
        pool = ResourcePool(["only"])
        async with await pool.acquire() as res:
            assert res == "only"

    @pytest.mark.asyncio
    async def test_resource_handle_aenter_multiple(self):
        pool = ResourcePool(["a", "b"])
        async with await pool.acquire(2) as resources:
            assert isinstance(resources, tuple)
            assert "a" in resources
            assert "b" in resources

    @pytest.mark.asyncio
    async def test_resource_handle_aexit_releases(self):
        pool = ResourcePool(["a", "b"])
        assert pool.available() == 2
        async with await pool.acquire(1):
            assert pool.available() == 1
        assert pool.available() == 2

    @pytest.mark.asyncio
    async def test_release_idempotent(self):
        pool = ResourcePool(["a"])
        handle = await pool.acquire(1)
        await handle.release()
        await handle.release()
        assert pool.available() == 1

    @pytest.mark.asyncio
    async def test_fifo_mode(self):
        pool = ResourcePool([1, 2, 3], stack_mode=False)
        h1 = await pool.acquire(1)
        async with h1:
            res1 = h1._resources[0]
        await h1.release()
        h2 = await pool.acquire(1)
        res2 = h2._resources[0]
        await h2.release()
        assert res1 == 1
        assert res2 == 2

    @pytest.mark.asyncio
    async def test_lifo_mode(self):
        pool = ResourcePool([1, 2, 3], stack_mode=True)
        h1 = await pool.acquire(1)
        res1 = h1._resources[0]
        await h1.release()
        h2 = await pool.acquire(1)
        res2 = h2._resources[0]
        await h2.release()
        assert res1 == res2

    def test_close_sync_callback(self):
        called = []
        pool = ResourcePool([1, 2])
        pool.on_close(lambda: called.append(True))
        pool.close()
        assert called == [True]

    def test_get_usage_distribution(self):
        pool = ResourcePool([1, 2])
        dist = pool.get_usage_distribution()
        assert isinstance(dist, str)
        assert "Resource" in dist

    @pytest.mark.asyncio
    async def test_release_resource_direct(self):
        pool = ResourcePool(["a", "b"])
        h = await pool.acquire(1)
        res = h._resources[0]
        await h.release()
        await pool.release_resource(res)
        assert pool.available() == 2


class TestAcquireFromMany:
    @pytest.mark.asyncio
    async def test_acquire_two_pools(self):
        p1 = ResourcePool(["a"])
        p2 = ResourcePool(["b"])
        handle = await acquire_from_many([p1, p2])
        async with handle as resources:
            assert len(resources) == 2
        assert p1.available() == 1
        assert p2.available() == 1

    @pytest.mark.asyncio
    async def test_acquire_three_pools(self):
        p1 = ResourcePool([1])
        p2 = ResourcePool([2])
        p3 = ResourcePool([3])
        counts = [1, 1, 1]
        handle = await acquire_from_many([p1, p2, p3], counts=counts)
        async with handle as resources:
            assert len(resources) == 3
        assert p1.available() == 1
        assert p2.available() == 1
        assert p3.available() == 1

    @pytest.mark.asyncio
    async def test_counts_mismatch_raises(self):
        p1 = ResourcePool([1])
        with pytest.raises(ValueError):
            await acquire_from_many([p1], counts=[1, 2])

    @pytest.mark.asyncio
    async def test_default_counts(self):
        p1 = ResourcePool([1])
        p2 = ResourcePool([2])
        handle = await acquire_from_many([p1, p2])
        assert handle is not None
        async with handle:
            pass

    @pytest.mark.asyncio
    async def test_concurrent_acquire_release(self):
        pool = ResourcePool([1, 2, 3])

        async def worker():
            async with await pool.acquire(1):
                await asyncio.sleep(0.02)

        await asyncio.gather(worker(), worker(), worker())
        assert pool.available() == 3
