"""
Unit tests for A* pathfinding algorithm.

Test coverage:
- Basic pathfinding with different heuristics
- Grid-based pathfinding
- Edge cases and error handling
- Heuristic functions
"""

import pytest
from algorithms.graphs.astar import (
    astar,
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    zero_heuristic
)


class TestAStarBasic:
    """Test basic A* pathfinding functionality."""

    def test_simple_path(self) -> None:
        """Test finding a simple path in a graph."""
        graph = {
            'A': [('B', 1), ('C', 4)],
            'B': [('C', 2), ('D', 5)],
            'C': [('D', 1)],
            'D': []
        }

        def simple_heuristic(a, b):
            # Simple heuristic: estimated distance to goal
            distances = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
            return distances.get(a, 0)

        path, cost = astar(graph, 'A', 'D', simple_heuristic)

        assert cost == 4  # A -> B -> C -> D
        assert path == ['A', 'B', 'C', 'D']

    def test_direct_path(self) -> None:
        """Test when direct path exists."""
        graph = {
            'A': [('B', 10), ('C', 1)],
            'B': [('D', 1)],
            'C': [('D', 1)],
            'D': []
        }

        path, cost = astar(graph, 'A', 'D', zero_heuristic)

        # Should take A -> C -> D (cost 2) not A -> B -> D (cost 11)
        assert cost == 2
        assert path == ['A', 'C', 'D']

    def test_start_equals_goal(self) -> None:
        """Test when start and goal are the same."""
        graph = {
            'A': [('B', 1)],
            'B': []
        }

        path, cost = astar(graph, 'A', 'A', zero_heuristic)

        assert cost == 0
        assert path == ['A']


class TestAStarGrid:
    """Test A* on grid-based graphs."""

    @staticmethod
    def create_grid(width: int, height: int, obstacles=None):
        """Helper to create a grid graph."""
        if obstacles is None:
            obstacles = set()

        graph = {}
        for x in range(width):
            for y in range(height):
                if (x, y) in obstacles:
                    continue

                neighbors = []
                # 4-directional movement
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if (nx, ny) not in obstacles:
                            neighbors.append(((nx, ny), 1))

                graph[(x, y)] = neighbors

        return graph

    def test_grid_manhattan(self) -> None:
        """Test A* on a grid with Manhattan distance."""
        grid = self.create_grid(5, 5)
        start = (0, 0)
        goal = (4, 4)

        path, cost = astar(grid, start, goal, manhattan_distance)

        assert cost == 8  # 4 right + 4 down
        assert len(path) == 9  # 9 cells including start and goal

    def test_grid_with_obstacles(self) -> None:
        """Test pathfinding around obstacles."""
        obstacles = {(2, i) for i in range(4)}  # Vertical wall
        grid = self.create_grid(5, 5, obstacles)

        start = (0, 2)
        goal = (4, 2)

        path, cost = astar(grid, start, goal, manhattan_distance)

        # Must go around the wall
        assert cost > 4  # Longer than direct path
        assert goal in path
        # Ensure path doesn't go through obstacles
        for node in path:
            assert node not in obstacles

    def test_grid_euclidean_vs_manhattan(self) -> None:
        """Compare Euclidean and Manhattan heuristics."""
        grid = self.create_grid(10, 10)
        start = (0, 0)
        goal = (9, 9)

        path_eucl, cost_eucl = astar(grid, start, goal, euclidean_distance)
        path_manh, cost_manh = astar(grid, start, goal, manhattan_distance)

        # Both should find optimal path (same cost)
        assert cost_eucl == cost_manh
        # Both paths should have same length
        assert len(path_eucl) == len(path_manh)


class TestAStarHeuristics:
    """Test different heuristic functions."""

    def test_manhattan_distance(self) -> None:
        """Test Manhattan distance calculation."""
        assert manhattan_distance((0, 0), (3, 4)) == 7
        assert manhattan_distance((1, 1), (1, 1)) == 0
        assert manhattan_distance((5, 5), (2, 3)) == 5

    def test_euclidean_distance(self) -> None:
        """Test Euclidean distance calculation."""
        assert euclidean_distance((0, 0), (3, 4)) == 5.0
        assert euclidean_distance((1, 1), (1, 1)) == 0.0
        assert abs(euclidean_distance((0, 0), (1, 1)) - 1.414) < 0.01

    def test_chebyshev_distance(self) -> None:
        """Test Chebyshev distance calculation."""
        assert chebyshev_distance((0, 0), (3, 4)) == 4
        assert chebyshev_distance((1, 1), (1, 1)) == 0
        assert chebyshev_distance((5, 5), (2, 3)) == 3

    def test_zero_heuristic(self) -> None:
        """Test zero heuristic (equivalent to Dijkstra)."""
        assert zero_heuristic('A', 'B') == 0
        assert zero_heuristic(1, 2) == 0
        assert zero_heuristic((0, 0), (10, 10)) == 0


class TestAStarErrors:
    """Test error handling."""

    def test_start_not_in_graph(self) -> None:
        """Test error when start node doesn't exist."""
        graph = {'A': [('B', 1)], 'B': []}

        with pytest.raises(ValueError, match="Start node 'Z' not found"):
            astar(graph, 'Z', 'B', zero_heuristic)

    def test_goal_not_in_graph(self) -> None:
        """Test error when goal node doesn't exist."""
        graph = {'A': [('B', 1)], 'B': []}

        with pytest.raises(ValueError, match="Goal node 'Z' not found"):
            astar(graph, 'A', 'Z', zero_heuristic)

    def test_no_path_exists(self) -> None:
        """Test error when no path exists between start and goal."""
        graph = {
            'A': [('B', 1)],
            'B': [],
            'C': [('D', 1)],
            'D': []
        }

        with pytest.raises(ValueError, match="No path exists"):
            astar(graph, 'A', 'D', zero_heuristic)


class TestAStarOptimality:
    """Test that A* finds optimal paths."""

    def test_chooses_shorter_path(self) -> None:
        """Test that A* chooses the shortest path when multiple exist."""
        graph = {
            'A': [('B', 1), ('C', 10)],
            'B': [('D', 1)],
            'C': [('D', 1)],
            'D': []
        }

        path, cost = astar(graph, 'A', 'D', zero_heuristic)

        assert cost == 2  # A -> B -> D, not A -> C -> D
        assert path == ['A', 'B', 'D']

    def test_admissible_heuristic_optimality(self) -> None:
        """Test that admissible heuristic guarantees optimal solution."""
        # Create a graph where heuristic might mislead greedy approach
        graph = {
            'A': [('B', 1), ('C', 2)],
            'B': [('D', 10)],
            'C': [('D', 1)],
            'D': []
        }

        def admissible_heuristic(node, goal):
            # Admissible: never overestimates
            estimates = {'A': 2, 'B': 1, 'C': 1, 'D': 0}
            return estimates.get(node, 0)

        path, cost = astar(graph, 'A', 'D', admissible_heuristic)

        # Should find A -> C -> D (cost 3), not A -> B -> D (cost 11)
        assert cost == 3
        assert path == ['A', 'C', 'D']


class TestAStarPerformance:
    """Test performance characteristics."""

    def test_large_grid(self) -> None:
        """Test A* on a larger grid."""
        def create_large_grid(size):
            graph = {}
            for x in range(size):
                for y in range(size):
                    neighbors = []
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            neighbors.append(((nx, ny), 1))
                    graph[(x, y)] = neighbors
            return graph

        grid = create_large_grid(20)
        start = (0, 0)
        goal = (19, 19)

        path, cost = astar(grid, start, goal, manhattan_distance)

        assert cost == 38  # 19 + 19
        assert len(path) == 39
        assert path[0] == start
        assert path[-1] == goal

    def test_early_termination(self) -> None:
        """Test that A* terminates upon reaching goal."""
        # A* should not explore all nodes
        graph = {}
        for i in range(100):
            graph[i] = [(i + 1, 1)] if i < 99 else []

        visited_count = [0]

        def counting_heuristic(node, goal):
            visited_count[0] += 1
            return abs(goal - node)

        path, cost = astar(graph, 0, 10, counting_heuristic)

        assert cost == 10
        assert len(path) == 11
        # Should not have visited all 100 nodes
        assert visited_count[0] < 100
