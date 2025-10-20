"""
0/1 Knapsack Problem.

The 0/1 knapsack problem is a classic dynamic programming problem where you need
to select items with given weights and values to maximize total value without
exceeding a weight capacity. Each item can be selected at most once (0 or 1).

Time Complexity: O(nW) where n is number of items and W is capacity
Space Complexity: O(nW) for standard DP, O(W) for space-optimized version

Example:
    >>> items = [
    ...     (10, 60),  # (weight, value)
    ...     (20, 100),
    ...     (30, 120)
    ... ]
    >>> max_value, selected = knapsack_01(items, capacity=50)
    >>> max_value
    220  # Select items 0 and 1
"""

from typing import List, Tuple


def knapsack_01(
    items: List[Tuple[int, int]],
    capacity: int
) -> Tuple[int, List[int]]:
    """
    Solve 0/1 knapsack problem using dynamic programming.

    Args:
        items: List of (weight, value) tuples for each item
        capacity: Maximum weight capacity of the knapsack

    Returns:
        A tuple containing:
        - max_value: Maximum value achievable
        - selected_items: Indices of items to select

    Raises:
        ValueError: If capacity is negative
        ValueError: If any item has negative weight or value

    Time Complexity: O(nW) where n is items count, W is capacity
    Space Complexity: O(nW) for the DP table

    Algorithm:
        Uses a 2D DP table where dp[i][w] represents the maximum value
        achievable using the first i items with weight limit w.

        Recurrence relation:
        dp[i][w] = max(
            dp[i-1][w],                    # Don't take item i
            dp[i-1][w-weight[i]] + value[i]  # Take item i
        )
    """
    if capacity < 0:
        raise ValueError(f"Capacity must be non-negative, got {capacity}")

    for idx, (weight, value) in enumerate(items):
        if weight < 0:
            raise ValueError(f"Item {idx} has negative weight: {weight}")
        if value < 0:
            raise ValueError(f"Item {idx} has negative value: {value}")

    if not items or capacity == 0:
        return 0, []

    n = len(items)

    # Create DP table: dp[i][w] = max value using first i items with capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Fill the DP table
    for i in range(1, n + 1):
        weight, value = items[i - 1]

        for w in range(capacity + 1):
            # Option 1: Don't take item i-1
            dp[i][w] = dp[i - 1][w]

            # Option 2: Take item i-1 (if it fits)
            if weight <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weight] + value)

    # Backtrack to find which items were selected
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        # If value changed, we took this item
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)  # Store 0-indexed item number
            weight, _ = items[i - 1]
            w -= weight

    selected.reverse()  # Reverse to get items in original order

    return dp[n][capacity], selected


def knapsack_01_space_optimized(
    items: List[Tuple[int, int]],
    capacity: int
) -> int:
    """
    Solve 0/1 knapsack with space optimization (only returns max value).

    This version uses O(W) space instead of O(nW) by only keeping
    the current and previous rows of the DP table.

    Args:
        items: List of (weight, value) tuples for each item
        capacity: Maximum weight capacity of the knapsack

    Returns:
        Maximum value achievable

    Raises:
        ValueError: If capacity is negative
        ValueError: If any item has negative weight or value

    Time Complexity: O(nW)
    Space Complexity: O(W)

    Note:
        This optimized version only computes the maximum value,
        not which items to select. Use the standard version if
        you need to know which items were chosen.
    """
    if capacity < 0:
        raise ValueError(f"Capacity must be non-negative, got {capacity}")

    for idx, (weight, value) in enumerate(items):
        if weight < 0:
            raise ValueError(f"Item {idx} has negative weight: {weight}")
        if value < 0:
            raise ValueError(f"Item {idx} has negative value: {value}")

    if not items or capacity == 0:
        return 0

    # Only need one array, updated in reverse to avoid overwriting needed values
    dp = [0] * (capacity + 1)

    for weight, value in items:
        # Traverse from right to left to avoid using updated values
        for w in range(capacity, weight - 1, -1):
            dp[w] = max(dp[w], dp[w - weight] + value)

    return dp[capacity]


def knapsack_unbounded(
    items: List[Tuple[int, int]],
    capacity: int
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Solve unbounded knapsack problem (can take unlimited copies of each item).

    Unlike 0/1 knapsack, each item can be selected multiple times.

    Args:
        items: List of (weight, value) tuples for each item
        capacity: Maximum weight capacity of the knapsack

    Returns:
        A tuple containing:
        - max_value: Maximum value achievable
        - selected_items: List of (item_index, count) tuples

    Raises:
        ValueError: If capacity is negative
        ValueError: If any item has negative or zero weight

    Time Complexity: O(nW)
    Space Complexity: O(W)

    Algorithm:
        dp[w] = max value achievable with capacity w
        dp[w] = max(dp[w], dp[w - weight[i]] + value[i]) for all items i
    """
    if capacity < 0:
        raise ValueError(f"Capacity must be non-negative, got {capacity}")

    for idx, (weight, value) in enumerate(items):
        if weight <= 0:
            raise ValueError(
                f"Item {idx} has non-positive weight: {weight}. "
                "Unbounded knapsack requires positive weights."
            )
        if value < 0:
            raise ValueError(f"Item {idx} has negative value: {value}")

    if not items or capacity == 0:
        return 0, []

    # dp[w] = (max_value, last_item_used)
    dp = [(0, -1) for _ in range(capacity + 1)]

    # Fill DP table
    for w in range(1, capacity + 1):
        for item_idx, (weight, value) in enumerate(items):
            if weight <= w:
                new_value = dp[w - weight][0] + value
                if new_value > dp[w][0]:
                    dp[w] = (new_value, item_idx)

    # Backtrack to find items
    selected = []
    w = capacity
    while w > 0 and dp[w][1] != -1:
        item_idx = dp[w][1]
        selected.append(item_idx)
        weight, _ = items[item_idx]
        w -= weight

    # Count occurrences of each item
    from collections import Counter
    item_counts = Counter(selected)
    selected_items = [(item, count) for item, count in sorted(item_counts.items())]

    return dp[capacity][0], selected_items


def knapsack_fractional(
    items: List[Tuple[int, int]],
    capacity: int
) -> Tuple[float, List[Tuple[int, float]]]:
    """
    Solve fractional knapsack problem (can take fractions of items).

    This is the greedy variant where items can be divided. Always optimal
    to take items in order of value-to-weight ratio.

    Args:
        items: List of (weight, value) tuples for each item
        capacity: Maximum weight capacity of the knapsack

    Returns:
        A tuple containing:
        - max_value: Maximum value achievable
        - selected_items: List of (item_index, fraction) tuples

    Raises:
        ValueError: If capacity is negative
        ValueError: If any item has non-positive weight or negative value

    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n) for storing items with ratios

    Algorithm:
        1. Calculate value/weight ratio for each item
        2. Sort items by ratio in descending order
        3. Greedily take items (or fractions) until capacity is full
    """
    if capacity < 0:
        raise ValueError(f"Capacity must be non-negative, got {capacity}")

    for idx, (weight, value) in enumerate(items):
        if weight <= 0:
            raise ValueError(f"Item {idx} has non-positive weight: {weight}")
        if value < 0:
            raise ValueError(f"Item {idx} has negative value: {value}")

    if not items or capacity == 0:
        return 0.0, []

    # Calculate value-to-weight ratios and sort
    items_with_ratio = [
        (idx, weight, value, value / weight)
        for idx, (weight, value) in enumerate(items)
    ]
    items_with_ratio.sort(key=lambda x: x[3], reverse=True)

    total_value = 0.0
    selected = []
    remaining_capacity = capacity

    for idx, weight, value, ratio in items_with_ratio:
        if remaining_capacity == 0:
            break

        if weight <= remaining_capacity:
            # Take whole item
            total_value += value
            selected.append((idx, 1.0))
            remaining_capacity -= weight
        else:
            # Take fraction of item
            fraction = remaining_capacity / weight
            total_value += value * fraction
            selected.append((idx, fraction))
            remaining_capacity = 0

    return total_value, selected


if __name__ == "__main__":
    print("Knapsack Problem Variations\n")

    # Test data
    items = [
        (10, 60),   # Item 0: weight=10, value=60
        (20, 100),  # Item 1: weight=20, value=100
        (30, 120),  # Item 2: weight=30, value=120
    ]
    capacity = 50

    print("Items:")
    for idx, (w, v) in enumerate(items):
        print(f"  Item {idx}: weight={w}, value={v}, ratio={v/w:.2f}")

    print(f"\nKnapsack capacity: {capacity}\n")

    # 0/1 Knapsack
    print("="*60)
    print("1. 0/1 Knapsack (each item 0 or 1 time)")
    max_val, selected = knapsack_01(items, capacity)
    print(f"Maximum value: {max_val}")
    print(f"Selected items: {selected}")
    print(f"Total weight: {sum(items[i][0] for i in selected)}")

    # Space-optimized
    max_val_opt = knapsack_01_space_optimized(items, capacity)
    print(f"\nSpace-optimized result: {max_val_opt}")

    # Unbounded Knapsack
    print("\n" + "="*60)
    print("2. Unbounded Knapsack (unlimited copies of each item)")
    max_val_unb, selected_unb = knapsack_unbounded(items, capacity)
    print(f"Maximum value: {max_val_unb}")
    print(f"Selected items (item_index, count): {selected_unb}")

    # Fractional Knapsack
    print("\n" + "="*60)
    print("3. Fractional Knapsack (can take fractions)")
    max_val_frac, selected_frac = knapsack_fractional(items, capacity)
    print(f"Maximum value: {max_val_frac:.2f}")
    print("Selected items (item_index, fraction):")
    for item_idx, fraction in selected_frac:
        weight, value = items[item_idx]
        print(f"  Item {item_idx}: {fraction*100:.1f}% "
              f"(weight={weight*fraction:.1f}, value={value*fraction:.1f})")
