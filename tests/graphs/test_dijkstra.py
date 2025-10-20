"""
Unit tests for Dijkstra's shortest path algorithm.

This test suite covers:
- Basic functionality with simple graphs
- Edge cases (single node, disconnected nodes, multiple paths)
- Error handling (invalid inputs, negative weights)
- Performance with larger graphs
"""

import pytest
from algorithms.graphs.dijkstra import dijkstra, dijkstra_single_target


class TestDijkstraBasic:
    """Test basic functionality of Dijkstra's algorithm."""

    def test_simple_graph(self) -> None:
        """Test with a simple connected graph."""
        graph = {
            'A': [('B', 4), ('C', 2)],
            'B': [('C', 1), ('D', 5)],
            'C': [('D', 8), ('E', 10)],
            'D': [('E', 2)],
            'E': []
        }

        distances, paths = dijkstra(graph, 'A')

        assert distances['A'] == 0
        assert distances['B'] == 4
        assert distances['C'] == 2
        assert distances['D'] == 9
        assert distances['E'] == 11

        assert paths['A'] == ['A']
        # Path to E should start with A and end with E, with correct length
        assert paths['E'][0] == 'A'
        assert paths['E'][-1] == 'E'
        assert len(paths['E']) == 5  # A -> B -> D -> E is one valid path

    def test_single_node(self) -> None:
        """Test with a graph containing a single node."""
        graph = {'A': []}
        distances, paths = dijkstra(graph, 'A')

        assert distances['A'] == 0
        assert paths['A'] == ['A']

    def test_two_nodes_connected(self) -> None:
        """Test with two connected nodes."""
        graph = {
            'A': [('B', 5)],
            'B': []
        }
        distances, paths = dijkstra(graph, 'A')

        assert distances['A'] == 0
        assert distances['B'] == 5
        assert paths['B'] == ['A', 'B']

    def test_disconnected_node(self) -> None:
        """Test with a disconnected node (unreachable)."""
        graph = {
            'A': [('B', 1)],
            'B': [],
            'C': [('D', 1)],
            'D': []
        }
        distances, paths = dijkstra(graph, 'A')

        assert distances['A'] == 0
        assert distances['B'] == 1
        assert distances['C'] == float('inf')
        assert distances['D'] == float('inf')
        assert paths['C'] == []
        assert paths['D'] == []


class TestDijkstraMultiplePaths:
    """Test behavior when multiple paths exist."""

    def test_finds_shortest_among_multiple_paths(self) -> None:
        """Test that Dijkstra finds the shortest path when multiple paths exist."""
        graph = {
            'A': [('B', 10), ('C', 3)],
            'B': [('D', 2)],
            'C': [('B', 4), ('D', 8)],
            'D': []
        }
        distances, paths = dijkstra(graph, 'A')

        # Shortest path to B is A -> C -> B (3 + 4 = 7), not A -> B (10)
        assert distances['B'] == 7
        assert paths['B'] == ['A', 'C', 'B']

        # Shortest path to D is A -> C -> B -> D (3 + 4 + 2 = 9), not A -> C -> D (11)
        assert distances['D'] == 9
        assert paths['D'] == ['A', 'C', 'B', 'D']


class TestDijkstraEdgeCases:
    """Test edge cases and special scenarios."""

    def test_zero_weight_edges(self) -> None:
        """Test with zero-weight edges."""
        graph = {
            'A': [('B', 0), ('C', 5)],
            'B': [('C', 3)],
            'C': []
        }
        distances, paths = dijkstra(graph, 'A')

        assert distances['B'] == 0
        assert distances['C'] == 3

    def test_self_loop_ignored(self) -> None:
        """Test that self-loops don't affect the result."""
        graph = {
            'A': [('A', 5), ('B', 2)],  # Self-loop on A
            'B': []
        }
        distances, paths = dijkstra(graph, 'A')

        assert distances['A'] == 0  # Self-loop shouldn't change this
        assert distances['B'] == 2

    def test_floating_point_weights(self) -> None:
        """Test with floating-point edge weights."""
        graph = {
            'A': [('B', 1.5), ('C', 2.7)],
            'B': [('C', 0.5)],
            'C': []
        }
        distances, paths = dijkstra(graph, 'A')

        assert distances['B'] == 1.5
        assert distances['C'] == 2.0  # Via B: 1.5 + 0.5


class TestDijkstraErrors:
    """Test error handling."""

    def test_start_not_in_graph(self) -> None:
        """Test that ValueError is raised if start vertex doesn't exist."""
        graph = {'A': [('B', 1)], 'B': []}

        with pytest.raises(ValueError, match="Start vertex 'Z' not found"):
            dijkstra(graph, 'Z')

    def test_negative_weight_raises_error(self) -> None:
        """Test that negative weights raise ValueError."""
        graph = {
            'A': [('B', -1)],  # Negative weight
            'B': []
        }

        with pytest.raises(ValueError, match="Negative edge weight"):
            dijkstra(graph, 'A')


class TestDijkstraSingleTarget:
    """Test the single-target optimization."""

    def test_single_target_basic(self) -> None:
        """Test finding path to a specific target."""
        graph = {
            'A': [('B', 4), ('C', 2)],
            'B': [('D', 5)],
            'C': [('D', 8)],
            'D': []
        }

        distance, path = dijkstra_single_target(graph, 'A', 'D')

        assert distance == 9
        assert path == ['A', 'B', 'D']

    def test_single_target_unreachable(self) -> None:
        """Test that ValueError is raised when target is unreachable."""
        graph = {
            'A': [('B', 1)],
            'B': [],
            'C': []  # Unreachable from A
        }

        with pytest.raises(ValueError, match="No path exists from 'A' to 'C'"):
            dijkstra_single_target(graph, 'A', 'C')

    def test_single_target_same_as_start(self) -> None:
        """Test when start and target are the same."""
        graph = {
            'A': [('B', 1)],
            'B': []
        }

        distance, path = dijkstra_single_target(graph, 'A', 'A')

        assert distance == 0
        assert path == ['A']

    def test_target_not_in_graph(self) -> None:
        """Test that ValueError is raised if target doesn't exist."""
        graph = {'A': [('B', 1)], 'B': []}

        with pytest.raises(ValueError, match="Target vertex 'Z' not found"):
            dijkstra_single_target(graph, 'A', 'Z')


class TestDijkstraPerformance:
    """Test performance characteristics with larger graphs."""

    def test_large_linear_graph(self) -> None:
        """Test with a large linear graph (chain)."""
        # Create a chain: A -> B -> C -> ... -> Z
        nodes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        graph = {nodes[i]: [(nodes[i + 1], 1)] for i in range(len(nodes) - 1)}
        graph[nodes[-1]] = []

        distances, paths = dijkstra(graph, 'A')

        # Distance to last node should be len(nodes) - 1
        assert distances['Z'] == 25
        assert len(paths['Z']) == 26

    def test_complete_graph_small(self) -> None:
        """Test with a small complete graph (all nodes connected)."""
        nodes = ['A', 'B', 'C', 'D']
        graph = {}

        for node in nodes:
            edges = []
            for other in nodes:
                if node != other:
                    # Weight is alphabetical distance
                    weight = abs(ord(node) - ord(other))
                    edges.append((other, weight))
            graph[node] = edges

        distances, paths = dijkstra(graph, 'A')

        # Direct connections should be shortest
        assert distances['B'] == 1
        assert distances['C'] == 2
        assert distances['D'] == 3
