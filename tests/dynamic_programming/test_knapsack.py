"""
Unit tests for Knapsack algorithms (0/1, Unbounded, Fractional).

Test coverage:
- 0/1 Knapsack with standard and space-optimized versions
- Unbounded Knapsack
- Fractional Knapsack
- Edge cases and error handling
"""

import pytest
from algorithms.dynamic_programming.knapsack import (
    knapsack_01,
    knapsack_01_space_optimized,
    knapsack_unbounded,
    knapsack_fractional
)


class TestKnapsack01:
    """Test 0/1 Knapsack algorithm."""

    def test_basic_case(self) -> None:
        """Test basic knapsack problem."""
        items = [(10, 60), (20, 100), (30, 120)]
        max_value, selected = knapsack_01(items, capacity=50)

        assert max_value == 220
        assert set(selected) == {0, 1}

    def test_single_item_fits(self) -> None:
        """Test with single item that fits."""
        items = [(10, 50)]
        max_value, selected = knapsack_01(items, capacity=20)

        assert max_value == 50
        assert selected == [0]

    def test_single_item_doesnt_fit(self) -> None:
        """Test with single item that doesn't fit."""
        items = [(30, 100)]
        max_value, selected = knapsack_01(items, capacity=20)

        assert max_value == 0
        assert selected == []

    def test_all_items_fit(self) -> None:
        """Test when all items can be taken."""
        items = [(1, 1), (2, 2), (3, 3)]
        max_value, selected = knapsack_01(items, capacity=10)

        assert max_value == 6
        assert set(selected) == {0, 1, 2}

    def test_no_items_fit(self) -> None:
        """Test when no items fit."""
        items = [(10, 50), (20, 100), (30, 150)]
        max_value, selected = knapsack_01(items, capacity=5)

        assert max_value == 0
        assert selected == []

    def test_empty_items(self) -> None:
        """Test with empty item list."""
        max_value, selected = knapsack_01([], capacity=50)

        assert max_value == 0
        assert selected == []

    def test_zero_capacity(self) -> None:
        """Test with zero capacity."""
        items = [(10, 60), (20, 100)]
        max_value, selected = knapsack_01(items, capacity=0)

        assert max_value == 0
        assert selected == []

    def test_negative_capacity_raises_error(self) -> None:
        """Test that negative capacity raises ValueError."""
        items = [(10, 60)]

        with pytest.raises(ValueError, match="Capacity must be non-negative"):
            knapsack_01(items, capacity=-10)

    def test_negative_weight_raises_error(self) -> None:
        """Test that negative weight raises ValueError."""
        items = [(-5, 60), (20, 100)]

        with pytest.raises(ValueError, match="negative weight"):
            knapsack_01(items, capacity=50)

    def test_negative_value_raises_error(self) -> None:
        """Test that negative value raises ValueError."""
        items = [(10, -60), (20, 100)]

        with pytest.raises(ValueError, match="negative value"):
            knapsack_01(items, capacity=50)

    def test_classic_example(self) -> None:
        """Test with classic knapsack example."""
        items = [
            (2, 12),  # Item 0
            (1, 10),  # Item 1
            (3, 20),  # Item 2
            (2, 15),  # Item 3
        ]
        max_value, selected = knapsack_01(items, capacity=5)

        assert max_value == 37  # Items 1, 2, 3


class TestKnapsack01SpaceOptimized:
    """Test space-optimized 0/1 Knapsack."""

    def test_matches_standard_version(self) -> None:
        """Test that optimized version matches standard."""
        items = [(10, 60), (20, 100), (30, 120)]
        capacity = 50

        standard_value, _ = knapsack_01(items, capacity)
        optimized_value = knapsack_01_space_optimized(items, capacity)

        assert standard_value == optimized_value

    def test_empty_items(self) -> None:
        """Test with empty item list."""
        assert knapsack_01_space_optimized([], capacity=50) == 0

    def test_zero_capacity(self) -> None:
        """Test with zero capacity."""
        items = [(10, 60), (20, 100)]
        assert knapsack_01_space_optimized(items, capacity=0) == 0

    def test_negative_capacity_raises_error(self) -> None:
        """Test that negative capacity raises ValueError."""
        items = [(10, 60)]

        with pytest.raises(ValueError):
            knapsack_01_space_optimized(items, capacity=-10)


class TestKnapsackUnbounded:
    """Test Unbounded Knapsack algorithm."""

    def test_basic_unbounded(self) -> None:
        """Test basic unbounded knapsack."""
        items = [(1, 10), (2, 15), (3, 40)]
        max_value, counts = knapsack_unbounded(items, capacity=5)

        # Can take multiple of same item
        assert max_value >= 50

    def test_single_item_repeated(self) -> None:
        """Test taking single item multiple times."""
        items = [(2, 10)]
        max_value, counts = knapsack_unbounded(items, capacity=10)

        assert max_value == 50
        assert counts[0] == 5

    def test_empty_items(self) -> None:
        """Test with empty item list."""
        max_value, counts = knapsack_unbounded([], capacity=50)

        assert max_value == 0
        assert counts == []

    def test_zero_capacity(self) -> None:
        """Test with zero capacity."""
        items = [(1, 10), (2, 15)]
        max_value, counts = knapsack_unbounded(items, capacity=0)

        assert max_value == 0

    def test_negative_capacity_raises_error(self) -> None:
        """Test that negative capacity raises ValueError."""
        items = [(1, 10)]

        with pytest.raises(ValueError):
            knapsack_unbounded(items, capacity=-10)


class TestKnapsackFractional:
    """Test Fractional Knapsack algorithm."""

    def test_basic_fractional(self) -> None:
        """Test basic fractional knapsack."""
        items = [(10, 60), (20, 100), (30, 120)]
        max_value, fractions = knapsack_fractional(items, capacity=50)

        # Should be able to achieve higher value than 0/1
        assert max_value == 240.0
        assert fractions[0] == 1.0  # Take all of item 0
        assert fractions[1] == 1.0  # Take all of item 1
        assert fractions[2] == 1.0  # Take all of item 2

    def test_partial_item(self) -> None:
        """Test taking partial item."""
        items = [(10, 60), (20, 100), (30, 120)]
        max_value, fractions = knapsack_fractional(items, capacity=25)

        # Should take all of highest value-to-weight ratio items
        assert max_value > 0
        assert 0.0 <= sum(fractions) <= 3.0

    def test_exact_capacity(self) -> None:
        """Test when items exactly fill capacity."""
        items = [(10, 60), (20, 100)]
        max_value, fractions = knapsack_fractional(items, capacity=30)

        assert max_value == 160.0
        assert fractions[0] == 1.0
        assert fractions[1] == 1.0

    def test_empty_items(self) -> None:
        """Test with empty item list."""
        max_value, fractions = knapsack_fractional([], capacity=50)

        assert max_value == 0.0
        assert fractions == []

    def test_zero_capacity(self) -> None:
        """Test with zero capacity."""
        items = [(10, 60), (20, 100)]
        max_value, fractions = knapsack_fractional(items, capacity=0)

        assert max_value == 0.0

    def test_negative_capacity_raises_error(self) -> None:
        """Test that negative capacity raises ValueError."""
        items = [(10, 60)]

        with pytest.raises(ValueError):
            knapsack_fractional(items, capacity=-10)

    def test_greedy_selection(self) -> None:
        """Test that fractional uses greedy approach correctly."""
        # Item 0: value/weight = 6.0
        # Item 1: value/weight = 5.0
        # Item 2: value/weight = 4.0
        items = [(10, 60), (20, 100), (30, 120)]
        max_value, fractions = knapsack_fractional(items, capacity=15)

        # Should take all of item 0 (best ratio) and part of item 1
        assert fractions[0] == 1.0
        assert 0.0 < fractions[1] <= 1.0


class TestKnapsackComparison:
    """Test comparing different knapsack variants."""

    def test_fractional_better_than_01(self) -> None:
        """Test that fractional can achieve better or equal value."""
        items = [(10, 60), (20, 100), (30, 120)]
        capacity = 35

        value_01, _ = knapsack_01(items, capacity)
        value_frac, _ = knapsack_fractional(items, capacity)

        # Fractional should be >= 0/1
        assert value_frac >= value_01

    def test_unbounded_better_than_01(self) -> None:
        """Test that unbounded can achieve better value with repetition."""
        items = [(2, 15)]  # High value-to-weight ratio
        capacity = 10

        value_01, _ = knapsack_01(items, capacity)
        value_unbounded, _ = knapsack_unbounded(items, capacity)

        # Unbounded should be better (can repeat)
        assert value_unbounded >= value_01

    def test_all_variants_same_for_exact_fit(self) -> None:
        """Test all variants give same result when all items fit."""
        items = [(5, 10), (10, 20)]
        capacity = 15

        value_01, _ = knapsack_01(items, capacity)
        value_01_opt = knapsack_01_space_optimized(items, capacity)
        value_frac, _ = knapsack_fractional(items, capacity)

        assert value_01 == value_01_opt == value_frac


class TestKnapsackEdgeCases:
    """Test edge cases for all knapsack variants."""

    def test_large_capacity(self) -> None:
        """Test with very large capacity."""
        items = [(1, 1), (2, 2), (3, 3)]
        max_value, selected = knapsack_01(items, capacity=10000)

        # All items should be taken
        assert len(selected) == 3

    def test_many_items(self) -> None:
        """Test with many items."""
        items = [(i, i * 10) for i in range(1, 51)]
        max_value, selected = knapsack_01(items, capacity=100)

        assert max_value > 0
        assert len(selected) > 0

    def test_zero_weight_item(self) -> None:
        """Test with zero weight item."""
        items = [(0, 100), (10, 50)]
        max_value, selected = knapsack_01(items, capacity=10)

        # Should take zero-weight item
        assert 0 in selected

    def test_zero_value_item(self) -> None:
        """Test with zero value item."""
        items = [(10, 0), (20, 100)]
        max_value, selected = knapsack_01(items, capacity=30)

        # Should not take zero-value item
        assert 0 not in selected
        assert max_value == 100
