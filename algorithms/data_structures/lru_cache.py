"""
LRU (Least Recently Used) Cache.

An LRU cache is a data structure that stores a limited number of items and evicts
the least recently used item when capacity is exceeded. Provides O(1) time
complexity for both get and put operations.

Time Complexity: O(1) for get and put
Space Complexity: O(capacity)

Example:
    >>> cache = LRUCache(capacity=2)
    >>> cache.put(1, "one")
    >>> cache.put(2, "two")
    >>> cache.get(1)  # returns "one"
    >>> cache.put(3, "three")  # evicts key 2
    >>> cache.get(2)  # returns None (evicted)
"""

from typing import Any, Optional, Dict


class DLinkedNode:
    """Doubly linked list node for LRU cache."""

    def __init__(self, key: Any = 0, value: Any = 0) -> None:
        """
        Initialize a doubly linked list node.

        Args:
            key: The key associated with this node
            value: The value stored in this node
        """
        self.key = key
        self.value = value
        self.prev: Optional['DLinkedNode'] = None
        self.next: Optional['DLinkedNode'] = None


class LRUCache:
    """
    LRU Cache implementation using HashMap + Doubly Linked List.

    The cache maintains a doubly linked list where:
    - Head is the most recently used item
    - Tail is the least recently used item

    A hashmap provides O(1) access to nodes in the list.

    Time Complexity:
        - get: O(1)
        - put: O(1)

    Space Complexity: O(capacity)

    Attributes:
        capacity: Maximum number of items the cache can hold
        size: Current number of items in the cache
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialize LRU cache with given capacity.

        Args:
            capacity: Maximum number of items the cache can hold

        Raises:
            ValueError: If capacity is less than 1

        Example:
            >>> cache = LRUCache(capacity=3)
        """
        if capacity < 1:
            raise ValueError(f"Capacity must be at least 1, got {capacity}")

        self.capacity = capacity
        self.size = 0

        # HashMap for O(1) access: key -> node
        self.cache: Dict[Any, DLinkedNode] = {}

        # Dummy head and tail for easier list manipulation
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: DLinkedNode) -> None:
        """
        Add node right after the head (most recently used position).

        Args:
            node: Node to add to head
        """
        node.prev = self.head
        node.next = self.head.next

        if self.head.next:
            self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: DLinkedNode) -> None:
        """
        Remove a node from the doubly linked list.

        Args:
            node: Node to remove
        """
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev

    def _move_to_head(self, node: DLinkedNode) -> None:
        """
        Move an existing node to the head (mark as most recently used).

        Args:
            node: Node to move
        """
        self._remove_node(node)
        self._add_to_head(node)

    def _remove_tail(self) -> DLinkedNode:
        """
        Remove and return the node before the tail (least recently used).

        Returns:
            The removed node
        """
        if self.tail.prev is None or self.tail.prev == self.head:
            raise RuntimeError("Cannot remove from empty cache")

        node = self.tail.prev
        self._remove_node(node)
        return node

    def get(self, key: Any) -> Optional[Any]:
        """
        Get value for a key, marking it as recently used.

        Args:
            key: Key to look up

        Returns:
            Value associated with key, or None if key doesn't exist

        Time Complexity: O(1)

        Example:
            >>> cache = LRUCache(2)
            >>> cache.put(1, "one")
            >>> cache.get(1)
            'one'
            >>> cache.get(2)
            None
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        # Move to head since it was just accessed (most recently used)
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any) -> None:
        """
        Put a key-value pair in the cache, evicting LRU item if at capacity.

        Args:
            key: Key to store
            value: Value to store

        Time Complexity: O(1)

        Example:
            >>> cache = LRUCache(2)
            >>> cache.put(1, "one")
            >>> cache.put(2, "two")
            >>> cache.put(3, "three")  # Evicts key 1
        """
        if key in self.cache:
            # Key exists, update value and move to head
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            # New key
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self._add_to_head(node)
            self.size += 1

            # Check if capacity exceeded
            if self.size > self.capacity:
                # Remove least recently used (tail)
                tail = self._remove_tail()
                del self.cache[tail.key]
                self.size -= 1

    def delete(self, key: Any) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: Key to delete

        Returns:
            True if key was found and deleted, False otherwise

        Time Complexity: O(1)

        Example:
            >>> cache = LRUCache(2)
            >>> cache.put(1, "one")
            >>> cache.delete(1)
            True
            >>> cache.delete(1)
            False
        """
        if key not in self.cache:
            return False

        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        self.size -= 1
        return True

    def clear(self) -> None:
        """
        Clear all items from the cache.

        Time Complexity: O(1)

        Example:
            >>> cache = LRUCache(2)
            >>> cache.put(1, "one")
            >>> cache.clear()
            >>> cache.get(1)
            None
        """
        self.cache.clear()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def __len__(self) -> int:
        """Return current number of items in cache."""
        return self.size

    def __contains__(self, key: Any) -> bool:
        """Check if key exists in cache."""
        return key in self.cache

    def keys(self) -> list:
        """
        Return list of keys in order from most to least recently used.

        Time Complexity: O(n)

        Returns:
            List of keys in MRU to LRU order
        """
        keys = []
        current = self.head.next
        while current and current != self.tail:
            keys.append(current.key)
            current = current.next
        return keys

    def __repr__(self) -> str:
        """String representation of cache."""
        items = []
        current = self.head.next
        while current and current != self.tail:
            items.append(f"{current.key}: {current.value}")
            current = current.next
        return f"LRUCache({self.capacity})[{', '.join(items)}]"


if __name__ == "__main__":
    print("LRU Cache Demonstration\n")

    # Create cache with capacity 3
    cache = LRUCache(capacity=3)
    print(f"Created cache with capacity 3\n")

    # Add items
    print("Adding items:")
    cache.put(1, "one")
    print(f"  put(1, 'one') -> {cache}")

    cache.put(2, "two")
    print(f"  put(2, 'two') -> {cache}")

    cache.put(3, "three")
    print(f"  put(3, 'three') -> {cache}")

    # Access item (moves to front)
    print(f"\nAccessing key 1:")
    val = cache.get(1)
    print(f"  get(1) -> '{val}'")
    print(f"  Cache state: {cache}")
    print(f"  Keys (MRU to LRU): {cache.keys()}")

    # Add new item, causing eviction
    print(f"\nAdding 4th item (capacity is 3):")
    cache.put(4, "four")
    print(f"  put(4, 'four') -> {cache}")
    print(f"  Key 2 evicted (was LRU)")

    # Try to access evicted item
    print(f"\nTrying to access evicted key:")
    val = cache.get(2)
    print(f"  get(2) -> {val}")

    # Update existing item
    print(f"\nUpdating existing key:")
    cache.put(1, "ONE")
    print(f"  put(1, 'ONE') -> {cache}")

    # Delete item
    print(f"\nDeleting key 3:")
    deleted = cache.delete(3)
    print(f"  delete(3) -> {deleted}")
    print(f"  Cache state: {cache}")

    # Clear cache
    print(f"\nClearing cache:")
    cache.clear()
    print(f"  clear() -> {cache}")
    print(f"  Size: {len(cache)}")

    # Demonstrate with different data types
    print("\n" + "="*60)
    print("Using cache with different data types:")
    cache2 = LRUCache(capacity=2)

    cache2.put("user:123", {"name": "Alice", "age": 30})
    cache2.put("user:456", {"name": "Bob", "age": 25})

    print(f"Stored user data: {cache2}")

    user = cache2.get("user:123")
    print(f"Retrieved user: {user}")
