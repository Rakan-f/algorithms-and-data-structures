"""
Unit tests for Bellman-Ford Shortest Path Algorithm.

Test coverage:
- Basic shortest path finding
- Negative weight edges
- Negative weight cycle detection
- Single target path finding
- Edge cases and error handling
"""

import pytest
from algorithms.graphs.bellman_ford import (
    bellman_ford,
    bellman_ford_single_target
)


class TestBellmanFordBasic:
    """Test basic Bellman-Ford functionality."""

    def test_simple_graph(self) -> None:
        """Test with simple graph."""
        edges = [
            ('A', 'B', 4),
            ('A', 'C', 2),
            ('B', 'C', 1),
            ('C', 'D', 2),
            ('B', 'D', 5)
        ]
        vertices = ['A', 'B', 'C', 'D']

        distances, paths = bellman_ford(edges, vertices, 'A')

        assert distances['A'] == 0
        assert distances['B'] == 4
        assert distances['C'] == 2
        assert distances['D'] == 4
        assert paths['A'] == ['A']
        assert paths['D'][-1] == 'D'

    def test_single_vertex(self) -> None:
        """Test with single vertex graph."""
        edges: list = []
        vertices = ['A']

        distances, paths = bellman_ford(edges, vertices, 'A')

        assert distances['A'] == 0
        assert paths['A'] == ['A']

    def test_disconnected_graph(self) -> None:
        """Test with disconnected vertices."""
        edges = [
            ('A', 'B', 1),
            ('C', 'D', 1)
        ]
        vertices = ['A', 'B', 'C', 'D']

        distances, paths = bellman_ford(edges, vertices, 'A')

        assert distances['A'] == 0
        assert distances['B'] == 1
        assert distances['C'] == float('inf')
        assert distances['D'] == float('inf')
        assert paths['C'] == []
        assert paths['D'] == []

    def test_no_outgoing_edges(self) -> None:
        """Test starting from vertex with no outgoing edges."""
        edges = [
            ('A', 'B', 1),
            ('B', 'C', 1)
        ]
        vertices = ['A', 'B', 'C']

        distances, paths = bellman_ford(edges, vertices, 'C')

        assert distances['C'] == 0
        assert distances['A'] == float('inf')
        assert distances['B'] == float('inf')


class TestBellmanFordNegativeWeights:
    """Test Bellman-Ford with negative edge weights."""

    def test_negative_edge_weights(self) -> None:
        """Test graph with negative edge weights."""
        edges = [
            ('A', 'B', 4),
            ('A', 'C', 2),
            ('B', 'C', -3),
            ('C', 'D', 2)
        ]
        vertices = ['A', 'B', 'C', 'D']

        distances, paths = bellman_ford(edges, vertices, 'A')

        assert distances['A'] == 0
        assert distances['B'] == 4
        assert distances['C'] == 1  # Through B with negative edge
        assert distances['D'] == 3

    def test_all_negative_weights(self) -> None:
        """Test graph with all negative weights."""
        edges = [
            ('A', 'B', -1),
            ('B', 'C', -2),
            ('C', 'D', -3)
        ]
        vertices = ['A', 'B', 'C', 'D']

        distances, paths = bellman_ford(edges, vertices, 'A')

        assert distances['A'] == 0
        assert distances['B'] == -1
        assert distances['C'] == -3
        assert distances['D'] == -6


class TestBellmanFordNegativeCycles:
    """Test negative weight cycle detection."""

    def test_negative_cycle_detection(self) -> None:
        """Test that negative cycle is detected."""
        edges = [
            ('A', 'B', 1),
            ('B', 'C', -3),
            ('C', 'A', 1)  # Creates negative cycle
        ]
        vertices = ['A', 'B', 'C']

        with pytest.raises(ValueError, match="Negative weight cycle detected"):
            bellman_ford(edges, vertices, 'A')

    def test_negative_cycle_not_reachable(self) -> None:
        """Test graph with unreachable negative cycle."""
        edges = [
            ('A', 'B', 1),
            ('C', 'D', 1),
            ('D', 'C', -3)  # Negative cycle between C and D
        ]
        vertices = ['A', 'B', 'C', 'D']

        # Should succeed as cycle is not reachable from A
        distances, paths = bellman_ford(edges, vertices, 'A')

        assert distances['A'] == 0
        assert distances['B'] == 1


class TestBellmanFordSingleTarget:
    """Test single target path finding."""

    def test_find_path_to_target(self) -> None:
        """Test finding path to specific target."""
        edges = [
            ('A', 'B', 4),
            ('A', 'C', 2),
            ('C', 'D', 2)
        ]
        vertices = ['A', 'B', 'C', 'D']

        distance, path = bellman_ford_single_target(edges, vertices, 'A', 'D')

        assert distance == 4
        assert path[0] == 'A'
        assert path[-1] == 'D'

    def test_target_unreachable(self) -> None:
        """Test when target is unreachable."""
        edges = [
            ('A', 'B', 1),
            ('C', 'D', 1)
        ]
        vertices = ['A', 'B', 'C', 'D']

        with pytest.raises(ValueError, match="No path exists"):
            bellman_ford_single_target(edges, vertices, 'A', 'D')

    def test_start_equals_target(self) -> None:
        """Test when start equals target."""
        edges = [('A', 'B', 1)]
        vertices = ['A', 'B']

        distance, path = bellman_ford_single_target(edges, vertices, 'A', 'A')

        assert distance == 0
        assert path == ['A']

    def test_invalid_target(self) -> None:
        """Test with invalid target vertex."""
        edges = [('A', 'B', 1)]
        vertices = ['A', 'B']

        with pytest.raises(ValueError, match="Target vertex.*not found"):
            bellman_ford_single_target(edges, vertices, 'A', 'Z')


class TestBellmanFordErrors:
    """Test error handling."""

    def test_invalid_start_vertex(self) -> None:
        """Test with start vertex not in graph."""
        edges = [('A', 'B', 1)]
        vertices = ['A', 'B']

        with pytest.raises(ValueError, match="Start vertex.*not found"):
            bellman_ford(edges, vertices, 'Z')

    def test_empty_graph(self) -> None:
        """Test with empty graph."""
        edges: list = []
        vertices: list = []

        with pytest.raises(ValueError):
            bellman_ford(edges, vertices, 'A')


class TestBellmanFordComparison:
    """Test Bellman-Ford against expected results."""

    def test_matches_dijkstra_for_positive_weights(self) -> None:
        """Test that results match Dijkstra for positive weights."""
        edges = [
            ('A', 'B', 4),
            ('A', 'C', 2),
            ('B', 'C', 1),
            ('C', 'D', 7),
            ('B', 'D', 3)
        ]
        vertices = ['A', 'B', 'C', 'D']

        distances, _ = bellman_ford(edges, vertices, 'A')

        # These are known shortest distances
        assert distances['A'] == 0
        assert distances['B'] == 4
        assert distances['C'] == 2
        assert distances['D'] == 7

    def test_handles_duplicate_edges(self) -> None:
        """Test with duplicate edges (takes minimum weight)."""
        edges = [
            ('A', 'B', 5),
            ('A', 'B', 3),  # Shorter path
            ('A', 'B', 7)
        ]
        vertices = ['A', 'B']

        distances, _ = bellman_ford(edges, vertices, 'A')

        assert distances['B'] == 3


class TestBellmanFordComplexGraphs:
    """Test with complex graph structures."""

    def test_complete_graph(self) -> None:
        """Test with complete graph."""
        edges = [
            ('A', 'B', 1), ('A', 'C', 4), ('A', 'D', 3),
            ('B', 'A', 1), ('B', 'C', 2), ('B', 'D', 5),
            ('C', 'A', 4), ('C', 'B', 2), ('C', 'D', 1),
            ('D', 'A', 3), ('D', 'B', 5), ('D', 'C', 1)
        ]
        vertices = ['A', 'B', 'C', 'D']

        distances, paths = bellman_ford(edges, vertices, 'A')

        assert distances['A'] == 0
        assert all(distances[v] < float('inf') for v in vertices)

    def test_long_path(self) -> None:
        """Test with long path."""
        vertices = [str(i) for i in range(10)]
        edges = [(str(i), str(i + 1), 1) for i in range(9)]

        distances, paths = bellman_ford(edges, vertices, '0')

        assert distances['0'] == 0
        assert distances['9'] == 9
        assert len(paths['9']) == 10

    def test_multiple_paths_same_length(self) -> None:
        """Test graph with multiple shortest paths."""
        edges = [
            ('A', 'B', 1),
            ('A', 'C', 1),
            ('B', 'D', 1),
            ('C', 'D', 1)
        ]
        vertices = ['A', 'B', 'C', 'D']

        distances, paths = bellman_ford(edges, vertices, 'A')

        # Both paths A->B->D and A->C->D have same length
        assert distances['D'] == 2
        assert len(paths['D']) == 3


class TestBellmanFordEdgeCases:
    """Test edge cases."""

    def test_self_loop_positive(self) -> None:
        """Test with positive weight self-loop."""
        edges = [
            ('A', 'A', 5),
            ('A', 'B', 1)
        ]
        vertices = ['A', 'B']

        distances, _ = bellman_ford(edges, vertices, 'A')

        assert distances['A'] == 0  # Self-loop doesn't help
        assert distances['B'] == 1

    def test_zero_weight_edges(self) -> None:
        """Test with zero weight edges."""
        edges = [
            ('A', 'B', 0),
            ('B', 'C', 0),
            ('C', 'D', 0)
        ]
        vertices = ['A', 'B', 'C', 'D']

        distances, _ = bellman_ford(edges, vertices, 'A')

        assert all(distances[v] == 0 for v in vertices)

    def test_large_weights(self) -> None:
        """Test with large weight values."""
        edges = [
            ('A', 'B', 1000000),
            ('B', 'C', 2000000),
            ('C', 'D', 3000000)
        ]
        vertices = ['A', 'B', 'C', 'D']

        distances, _ = bellman_ford(edges, vertices, 'A')

        assert distances['D'] == 6000000


class TestBellmanFordApplications:
    """Test real-world applications."""

    def test_currency_arbitrage_detection(self) -> None:
        """Test detecting currency arbitrage (negative cycle)."""
        # Exchange rates where cycle creates profit
        edges = [
            ('USD', 'EUR', -0.1),  # Negative log of exchange rate
            ('EUR', 'GBP', -0.1),
            ('GBP', 'USD', -0.1)  # If sum < 0, arbitrage exists
        ]
        vertices = ['USD', 'EUR', 'GBP']

        with pytest.raises(ValueError, match="Negative weight cycle"):
            bellman_ford(edges, vertices, 'USD')

    def test_network_routing_with_costs(self) -> None:
        """Test network routing with mixed costs."""
        edges = [
            ('Router1', 'Router2', 10),
            ('Router1', 'Router3', 5),
            ('Router2', 'Router4', 1),
            ('Router3', 'Router4', 15),
            ('Router3', 'Router2', -3)  # Discount route
        ]
        vertices = ['Router1', 'Router2', 'Router3', 'Router4']

        distances, paths = bellman_ford(edges, vertices, 'Router1')

        assert distances['Router4'] == 3  # Router1->Router3->Router2->Router4
