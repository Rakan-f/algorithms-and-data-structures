"""
A* Pathfinding Algorithm.

A* (A-star) is an informed search algorithm that finds the shortest path between
a start and target node. It uses a heuristic function to guide the search,
making it more efficient than Dijkstra's algorithm when a good heuristic is available.

Time Complexity: O(E) in worst case, often much better with good heuristics
Space Complexity: O(V)

Example:
    >>> # Manhattan distance heuristic for grid pathfinding
    >>> def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    ...     return abs(a[0] - b[0]) + abs(a[1] - b[1])
    >>>
    >>> # Grid represented as graph
    >>> graph = {
    ...     (0, 0): [((0, 1), 1), ((1, 0), 1)],
    ...     (0, 1): [((0, 0), 1), ((1, 1), 1)],
    ...     ...
    ... }
    >>> path, cost = astar(graph, (0, 0), (5, 5), manhattan_distance)
"""

import heapq
from typing import Any, Callable, Dict, List, Tuple, Optional, Set


def astar(
    graph: Dict[Any, List[Tuple[Any, float]]],
    start: Any,
    goal: Any,
    heuristic: Callable[[Any, Any], float]
) -> Tuple[List[Any], float]:
    """
    Find shortest path from start to goal using A* algorithm.

    Args:
        graph: Adjacency list where graph[node] = [(neighbor, cost), ...]
        start: Starting node
        goal: Goal node
        heuristic: Function that estimates cost from a node to goal.
                   Must be admissible (never overestimate) for optimal results.

    Returns:
        A tuple containing:
        - path: List of nodes forming the shortest path from start to goal
        - cost: Total cost of the path

    Raises:
        ValueError: If start or goal not in graph
        ValueError: If no path exists from start to goal

    Time Complexity: O(E) worst case, often better with good heuristic
    Space Complexity: O(V) for storing open and closed sets

    Note:
        The heuristic function must be:
        1. Admissible: h(n) ≤ actual cost from n to goal
        2. Consistent (optional): h(n) ≤ cost(n, n') + h(n') for all neighbors n'

        Common heuristics:
        - Euclidean distance for spatial graphs
        - Manhattan distance for grid-based movement
        - Zero function (reduces to Dijkstra's algorithm)
    """
    if start not in graph:
        raise ValueError(f"Start node '{start}' not found in graph")
    if goal not in graph:
        raise ValueError(f"Goal node '{goal}' not found in graph")

    # g_score: actual cost from start to node
    g_score: Dict[Any, float] = {node: float('inf') for node in graph}
    g_score[start] = 0

    # f_score: g_score + heuristic (estimated total cost)
    f_score: Dict[Any, float] = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal)

    # Priority queue: (f_score, node)
    open_set: List[Tuple[float, Any]] = [(f_score[start], start)]

    # Track which nodes we've expanded to avoid reprocessing
    closed_set: Set[Any] = set()

    # Track the path
    came_from: Dict[Any, Any] = {}

    while open_set:
        current_f, current = heapq.heappop(open_set)

        # Skip outdated entries in the priority queue
        if current in closed_set:
            continue

        # Goal reached!
        if current == goal:
            return _reconstruct_path(came_from, current), g_score[current]

        closed_set.add(current)

        # Skip if this is an outdated entry
        if current_f > f_score[current]:
            continue

        # Explore neighbors
        for neighbor, cost in graph[current]:
            if neighbor in closed_set:
                continue

            # Calculate tentative g_score through current node
            tentative_g_score = g_score[current] + cost

            # Found a better path to neighbor
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # No path found
    raise ValueError(f"No path exists from '{start}' to '{goal}'")


def _reconstruct_path(came_from: Dict[Any, Any], current: Any) -> List[Any]:
    """
    Reconstruct the path from start to current using the came_from map.

    Args:
        came_from: Dictionary mapping each node to its predecessor
        current: Current (goal) node

    Returns:
        List of nodes from start to current
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two 2D points.

    This is an admissible heuristic for spatial graphs where diagonal movement
    is allowed and has the same cost as straight movement.

    Args:
        a: First point (x, y)
        b: Second point (x, y)

    Returns:
        Euclidean distance between points

    Time Complexity: O(1)
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Calculate Manhattan distance between two 2D grid points.

    This is an admissible heuristic for grid-based pathfinding where
    only horizontal and vertical movement is allowed.

    Args:
        a: First point (x, y)
        b: Second point (x, y)

    Returns:
        Manhattan distance between points

    Time Complexity: O(1)
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Calculate Chebyshev distance between two 2D grid points.

    This is an admissible heuristic for grid-based pathfinding where
    8-directional movement (including diagonals) is allowed with uniform cost.

    Args:
        a: First point (x, y)
        b: Second point (x, y)

    Returns:
        Chebyshev distance between points

    Time Complexity: O(1)
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def zero_heuristic(a: Any, b: Any) -> float:
    """
    Zero heuristic (reduces A* to Dijkstra's algorithm).

    This always returns 0, making A* equivalent to Dijkstra's algorithm.
    Useful for comparison or when no good heuristic is available.

    Args:
        a: First node (unused)
        b: Second node (unused)

    Returns:
        Always 0

    Time Complexity: O(1)
    """
    return 0.0


if __name__ == "__main__":
    # Example: Finding path in a 2D grid
    print("A* Pathfinding Example: 2D Grid\n")

    # Create a simple 5x5 grid graph
    # Nodes are (x, y) tuples
    def create_grid_graph(width: int, height: int) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
        """Create a grid graph with 4-directional movement."""
        graph = {}
        for x in range(width):
            for y in range(height):
                neighbors = []
                # Add neighbors (up, down, left, right)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbors.append(((nx, ny), 1.0))  # Cost of 1 per move
                graph[(x, y)] = neighbors
        return graph

    grid = create_grid_graph(5, 5)
    start_pos = (0, 0)
    goal_pos = (4, 4)

    print(f"Finding path from {start_pos} to {goal_pos}")
    print(f"Grid size: 5x5\n")

    # Compare different heuristics
    heuristics = [
        ("Manhattan Distance", manhattan_distance),
        ("Euclidean Distance", euclidean_distance),
        ("Zero Heuristic (Dijkstra)", zero_heuristic)
    ]

    for name, heuristic_func in heuristics:
        path, cost = astar(grid, start_pos, goal_pos, heuristic_func)
        print(f"{name}:")
        print(f"  Path length: {len(path)}")
        print(f"  Total cost: {cost}")
        print(f"  Path: {' -> '.join(str(p) for p in path)}")
        print()
