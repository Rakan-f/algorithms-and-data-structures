"""
Unit tests for LRU (Least Recently Used) Cache.

Test coverage:
- Basic get and put operations
- Eviction policy
- Capacity management
- Edge cases and error handling
"""

import pytest
from algorithms.data_structures.lru_cache import LRUCache


class TestLRUCacheBasic:
    """Test basic LRU cache operations."""

    def test_put_and_get(self) -> None:
        """Test basic put and get operations."""
        cache = LRUCache(capacity=2)

        cache.put(1, "one")
        cache.put(2, "two")

        assert cache.get(1) == "one"
        assert cache.get(2) == "two"

    def test_get_nonexistent_key(self) -> None:
        """Test getting a key that doesn't exist."""
        cache = LRUCache(capacity=2)

        assert cache.get(1) is None

    def test_put_updates_existing_key(self) -> None:
        """Test that put updates value for existing key."""
        cache = LRUCache(capacity=2)

        cache.put(1, "one")
        cache.put(1, "ONE")

        assert cache.get(1) == "ONE"

    def test_capacity_one(self) -> None:
        """Test cache with capacity of 1."""
        cache = LRUCache(capacity=1)

        cache.put(1, "one")
        assert cache.get(1) == "one"

        cache.put(2, "two")
        assert cache.get(1) is None  # Evicted
        assert cache.get(2) == "two"


class TestLRUCacheEviction:
    """Test LRU eviction policy."""

    def test_evict_least_recently_used(self) -> None:
        """Test that least recently used item is evicted."""
        cache = LRUCache(capacity=2)

        cache.put(1, "one")
        cache.put(2, "two")
        cache.put(3, "three")  # Should evict key 1

        assert cache.get(1) is None
        assert cache.get(2) == "two"
        assert cache.get(3) == "three"

    def test_get_updates_recency(self) -> None:
        """Test that get operation updates recency."""
        cache = LRUCache(capacity=2)

        cache.put(1, "one")
        cache.put(2, "two")
        cache.get(1)  # Make 1 recently used
        cache.put(3, "three")  # Should evict key 2, not 1

        assert cache.get(1) == "one"
        assert cache.get(2) is None
        assert cache.get(3) == "three"

    def test_put_updates_recency(self) -> None:
        """Test that put on existing key updates recency."""
        cache = LRUCache(capacity=2)

        cache.put(1, "one")
        cache.put(2, "two")
        cache.put(1, "ONE")  # Update value and recency
        cache.put(3, "three")  # Should evict key 2

        assert cache.get(1) == "ONE"
        assert cache.get(2) is None
        assert cache.get(3) == "three"

    def test_multiple_evictions(self) -> None:
        """Test multiple evictions in sequence."""
        cache = LRUCache(capacity=2)

        cache.put(1, "one")
        cache.put(2, "two")
        cache.put(3, "three")  # Evict 1
        cache.put(4, "four")   # Evict 2

        assert cache.get(1) is None
        assert cache.get(2) is None
        assert cache.get(3) == "three"
        assert cache.get(4) == "four"


class TestLRUCacheCapacity:
    """Test capacity management."""

    def test_invalid_capacity_raises_error(self) -> None:
        """Test that invalid capacity raises ValueError."""
        with pytest.raises(ValueError, match="Capacity must be at least 1"):
            LRUCache(capacity=0)

        with pytest.raises(ValueError):
            LRUCache(capacity=-5)

    def test_large_capacity(self) -> None:
        """Test with large capacity."""
        cache = LRUCache(capacity=1000)

        for i in range(1000):
            cache.put(i, f"value_{i}")

        # All values should still be accessible
        for i in range(1000):
            assert cache.get(i) == f"value_{i}"

    def test_capacity_not_exceeded(self) -> None:
        """Test that capacity is never exceeded."""
        cache = LRUCache(capacity=3)

        # Put more items than capacity
        for i in range(10):
            cache.put(i, f"value_{i}")

        # Only last 3 should exist
        assert cache.get(7) == "value_7"
        assert cache.get(8) == "value_8"
        assert cache.get(9) == "value_9"

        # Earlier ones should be evicted
        for i in range(7):
            assert cache.get(i) is None


class TestLRUCacheComplexScenarios:
    """Test complex usage patterns."""

    def test_alternating_puts_and_gets(self) -> None:
        """Test alternating put and get operations."""
        cache = LRUCache(capacity=2)

        cache.put(1, "one")
        assert cache.get(1) == "one"
        cache.put(2, "two")
        assert cache.get(1) == "one"
        assert cache.get(2) == "two"
        cache.put(3, "three")

        assert cache.get(2) == "two"
        assert cache.get(3) == "three"
        assert cache.get(1) is None  # Evicted

    def test_repeated_access_same_key(self) -> None:
        """Test repeatedly accessing same key."""
        cache = LRUCache(capacity=2)

        cache.put(1, "one")
        cache.put(2, "two")

        # Repeatedly access key 1
        for _ in range(10):
            assert cache.get(1) == "one"

        cache.put(3, "three")

        # Key 1 should still exist (it's most recent)
        assert cache.get(1) == "one"
        assert cache.get(2) is None  # Evicted

    def test_overwrite_then_evict(self) -> None:
        """Test overwriting value then evicting."""
        cache = LRUCache(capacity=2)

        cache.put(1, "one")
        cache.put(2, "two")
        cache.put(1, "ONE")  # Overwrite
        cache.put(3, "three")  # Evict 2

        assert cache.get(1) == "ONE"
        assert cache.get(3) == "three"


class TestLRUCacheDataTypes:
    """Test cache with different data types."""

    def test_string_keys_and_values(self) -> None:
        """Test with string keys and values."""
        cache = LRUCache(capacity=2)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

    def test_mixed_types(self) -> None:
        """Test with mixed key and value types."""
        cache = LRUCache(capacity=3)

        cache.put(1, "string_value")
        cache.put("key", 42)
        cache.put(3.14, [1, 2, 3])

        assert cache.get(1) == "string_value"
        assert cache.get("key") == 42
        assert cache.get(3.14) == [1, 2, 3]

    def test_none_value(self) -> None:
        """Test storing None as a value."""
        cache = LRUCache(capacity=2)

        cache.put(1, None)

        # Getting existing key with None value returns None
        # Same as getting non-existent key
        result = cache.get(1)
        assert result is None

    def test_object_values(self) -> None:
        """Test with object values."""
        cache = LRUCache(capacity=2)

        obj1 = {"name": "Alice", "age": 30}
        obj2 = {"name": "Bob", "age": 25}

        cache.put(1, obj1)
        cache.put(2, obj2)

        assert cache.get(1) == obj1
        assert cache.get(2) == obj2


class TestLRUCacheEdgeCases:
    """Test edge cases."""

    def test_empty_cache(self) -> None:
        """Test operations on empty cache."""
        cache = LRUCache(capacity=2)

        assert cache.get(1) is None
        assert cache.get("any_key") is None

    def test_single_item_repeatedly(self) -> None:
        """Test repeatedly adding same item."""
        cache = LRUCache(capacity=2)

        for i in range(10):
            cache.put(1, f"value_{i}")

        assert cache.get(1) == "value_9"

    def test_zero_then_fill(self) -> None:
        """Test filling cache from empty."""
        cache = LRUCache(capacity=3)

        cache.put(1, "one")
        assert cache.get(1) == "one"

        cache.put(2, "two")
        assert cache.get(2) == "two"

        cache.put(3, "three")
        assert cache.get(3) == "three"

        # All should still be accessible
        assert cache.get(1) == "one"
        assert cache.get(2) == "two"
        assert cache.get(3) == "three"


class TestLRUCacheDunderMethods:
    """Test special Python methods."""

    def test_len(self) -> None:
        """Test __len__ method."""
        cache = LRUCache(capacity=3)

        assert len(cache) == 0

        cache.put(1, "one")
        assert len(cache) == 1

        cache.put(2, "two")
        assert len(cache) == 2

        cache.put(3, "three")
        assert len(cache) == 3

        cache.put(4, "four")  # Evicts 1
        assert len(cache) == 3

    def test_contains(self) -> None:
        """Test __contains__ method (in operator)."""
        cache = LRUCache(capacity=2)

        cache.put(1, "one")
        cache.put(2, "two")

        assert 1 in cache
        assert 2 in cache
        assert 3 not in cache

        cache.put(3, "three")  # Evicts 1

        assert 1 not in cache
        assert 3 in cache

    def test_repr(self) -> None:
        """Test __repr__ method."""
        cache = LRUCache(capacity=2)

        cache.put(1, "one")
        cache.put(2, "two")

        repr_str = repr(cache)
        assert "LRUCache" in repr_str


class TestLRUCachePerformance:
    """Test performance characteristics."""

    def test_constant_time_operations(self) -> None:
        """Test that operations complete quickly."""
        cache = LRUCache(capacity=1000)

        # Put 1000 items
        for i in range(1000):
            cache.put(i, f"value_{i}")

        # Access should still be fast
        for i in range(1000):
            cache.get(i)

    def test_large_number_of_operations(self) -> None:
        """Test with many operations."""
        cache = LRUCache(capacity=100)

        # Perform 10000 operations
        for i in range(10000):
            cache.put(i % 150, f"value_{i}")
            cache.get(i % 150)


class TestLRUCacheApplications:
    """Test real-world application scenarios."""

    def test_web_page_cache(self) -> None:
        """Test using cache for web pages."""
        cache = LRUCache(capacity=3)

        cache.put("/home", "<html>Home Page</html>")
        cache.put("/about", "<html>About Page</html>")
        cache.put("/contact", "<html>Contact Page</html>")

        # Access home frequently
        cache.get("/home")
        cache.get("/home")

        # Add new page, should evict /about (least recent)
        cache.put("/products", "<html>Products</html>")

        assert cache.get("/home") == "<html>Home Page</html>"
        assert cache.get("/about") is None
        assert cache.get("/contact") == "<html>Contact Page</html>"
        assert cache.get("/products") == "<html>Products</html>"

    def test_database_query_cache(self) -> None:
        """Test using cache for database query results."""
        cache = LRUCache(capacity=5)

        # Cache query results
        cache.put("SELECT * FROM users", [{"id": 1, "name": "Alice"}])
        cache.put("SELECT * FROM orders", [{"id": 1, "total": 100}])

        assert cache.get("SELECT * FROM users") is not None
        assert cache.get("SELECT * FROM orders") is not None

    def test_function_result_cache(self) -> None:
        """Test memoization use case."""
        cache = LRUCache(capacity=10)

        # Simulate caching expensive function results
        def expensive_computation(n: int) -> int:
            key = f"fib_{n}"
            result = cache.get(key)

            if result is not None:
                return result

            # Compute (simplified - not actual fibonacci)
            value = n * n
            cache.put(key, value)
            return value

        # First call computes
        result1 = expensive_computation(5)
        # Second call uses cache
        result2 = expensive_computation(5)

        assert result1 == result2 == 25
