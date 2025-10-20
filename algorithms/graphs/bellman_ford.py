"""
Bellman-Ford Shortest Path Algorithm.

The Bellman-Ford algorithm finds shortest paths from a source vertex to all other
vertices in a weighted graph. Unlike Dijkstra's algorithm, it can handle negative
edge weights and detect negative weight cycles.

Time Complexity: O(VE)
Space Complexity: O(V)

Example:
    >>> graph = {
    ...     'edges': [
    ...         ('A', 'B', 4),
    ...         ('A', 'C', 2),
    ...         ('B', 'C', -3),
    ...         ('C', 'D', 2)
    ...     ],
    ...     'vertices': ['A', 'B', 'C', 'D']
    ... }
    >>> distances, paths = bellman_ford(graph, 'A')
    >>> distances['D']
    1
"""

from typing import Any, Dict, List, Tuple, Optional


def bellman_ford(
    edges: List[Tuple[Any, Any, float]],
    vertices: List[Any],
    start: Any
) -> Tuple[Dict[Any, float], Dict[Any, List[Any]]]:
    """
    Find shortest paths from start vertex using Bellman-Ford algorithm.

    This algorithm works with negative edge weights and can detect negative cycles.
    It relaxes all edges V-1 times to find shortest paths.

    Args:
        edges: List of tuples (source, destination, weight)
        vertices: List of all vertices in the graph
        start: Starting vertex

    Returns:
        A tuple containing:
        - distances: Dictionary mapping each vertex to its shortest distance from start
        - paths: Dictionary mapping each vertex to the shortest path from start

    Raises:
        ValueError: If start vertex is not in the graph
        ValueError: If a negative weight cycle is detected

    Time Complexity: O(VE) where V is vertices and E is edges
    Space Complexity: O(V) for storing distances and paths

    Note:
        Negative weight cycles make shortest paths undefined, as you can always
        achieve a shorter path by traversing the cycle again. If detected,
        this function raises a ValueError.
    """
    if start not in vertices:
        raise ValueError(f"Start vertex '{start}' not found in graph")

    # Initialize distances with infinity
    distances: Dict[Any, float] = {v: float('inf') for v in vertices}
    distances[start] = 0

    # Initialize paths
    paths: Dict[Any, List[Any]] = {v: [] for v in vertices}
    paths[start] = [start]

    # Relax edges V-1 times
    # After k iterations, we have correct shortest paths with at most k edges
    for _ in range(len(vertices) - 1):
        updated = False

        for source, dest, weight in edges:
            # Skip if source is unreachable
            if distances[source] == float('inf'):
                continue

            # Calculate distance through this edge
            new_distance = distances[source] + weight

            # If we found a shorter path, update it
            if new_distance < distances[dest]:
                distances[dest] = new_distance
                paths[dest] = paths[source] + [dest]
                updated = True

        # Optimization: if no updates occurred, we're done early
        if not updated:
            break

    # Check for negative weight cycles
    # If we can still relax edges, there's a negative cycle
    for source, dest, weight in edges:
        if distances[source] != float('inf'):
            if distances[source] + weight < distances[dest]:
                raise ValueError(
                    f"Negative weight cycle detected involving edge "
                    f"({source}, {dest}, {weight}). "
                    "Shortest paths are undefined in graphs with negative cycles."
                )

    return distances, paths


def bellman_ford_single_target(
    edges: List[Tuple[Any, Any, float]],
    vertices: List[Any],
    start: Any,
    target: Any
) -> Tuple[float, List[Any]]:
    """
    Find shortest path from start to target using Bellman-Ford.

    This is a wrapper around bellman_ford that returns only the path to a
    specific target. Unlike A* or Dijkstra, Bellman-Ford doesn't benefit
    from early termination, so it computes all shortest paths anyway.

    Args:
        edges: List of tuples (source, destination, weight)
        vertices: List of all vertices in the graph
        start: Starting vertex
        target: Target vertex

    Returns:
        A tuple containing:
        - distance: Shortest distance from start to target
        - path: Shortest path from start to target

    Raises:
        ValueError: If start or target not in graph
        ValueError: If negative weight cycle detected
        ValueError: If no path exists from start to target

    Time Complexity: O(VE)
    Space Complexity: O(V)
    """
    if target not in vertices:
        raise ValueError(f"Target vertex '{target}' not found in graph")

    distances, paths = bellman_ford(edges, vertices, start)

    if distances[target] == float('inf'):
        raise ValueError(f"No path exists from '{start}' to '{target}'")

    return distances[target], paths[target]


def detect_negative_cycle(
    edges: List[Tuple[Any, Any, float]],
    vertices: List[Any]
) -> Tuple[bool, Optional[List[Any]]]:
    """
    Detect if a negative weight cycle exists in the graph.

    This runs Bellman-Ford from an arbitrary vertex and checks if any
    edge can still be relaxed after V-1 iterations.

    Args:
        edges: List of tuples (source, destination, weight)
        vertices: List of all vertices in the graph

    Returns:
        A tuple containing:
        - has_cycle: True if negative cycle exists, False otherwise
        - cycle: List of vertices forming the cycle, or None if no cycle

    Time Complexity: O(VE)
    Space Complexity: O(V)

    Example:
        >>> edges = [('A', 'B', 1), ('B', 'C', -3), ('C', 'A', 1)]
        >>> vertices = ['A', 'B', 'C']
        >>> has_cycle, cycle = detect_negative_cycle(edges, vertices)
        >>> has_cycle
        True
        >>> cycle
        ['A', 'B', 'C', 'A']
    """
    if not vertices:
        return False, None

    # Run Bellman-Ford from first vertex
    start = vertices[0]
    distances: Dict[Any, float] = {v: float('inf') for v in vertices}
    distances[start] = 0
    parent: Dict[Any, Optional[Any]] = {v: None for v in vertices}

    # Relax edges V-1 times
    for _ in range(len(vertices) - 1):
        for source, dest, weight in edges:
            if distances[source] != float('inf'):
                if distances[source] + weight < distances[dest]:
                    distances[dest] = distances[source] + weight
                    parent[dest] = source

    # Check for negative cycle and find a vertex in it
    cycle_vertex = None
    for source, dest, weight in edges:
        if distances[source] != float('inf'):
            if distances[source] + weight < distances[dest]:
                # Found a vertex that's part of a negative cycle
                cycle_vertex = dest
                parent[dest] = source
                break

    if cycle_vertex is None:
        return False, None

    # Trace back to find the actual cycle
    # Move back V steps to ensure we're in the cycle
    for _ in range(len(vertices)):
        cycle_vertex = parent[cycle_vertex]

    # Extract the cycle
    cycle = []
    current = cycle_vertex
    while True:
        cycle.append(current)
        current = parent[current]
        if current == cycle_vertex:
            cycle.append(current)
            break

    return True, list(reversed(cycle))


if __name__ == "__main__":
    print("Bellman-Ford Algorithm Demonstration\n")

    # Example 1: Graph with negative weights (but no negative cycle)
    print("Example 1: Graph with negative edges")
    edges1 = [
        ('A', 'B', 4),
        ('A', 'C', 2),
        ('B', 'C', -3),  # Negative edge
        ('C', 'D', 2),
        ('B', 'D', 5)
    ]
    vertices1 = ['A', 'B', 'C', 'D']

    distances, paths = bellman_ford(edges1, vertices1, 'A')

    print("Shortest distances from A:")
    for v in sorted(distances.keys()):
        print(f"  A -> {v}: {distances[v]}")

    print("\nShortest paths:")
    for v in sorted(paths.keys()):
        if paths[v]:
            print(f"  {' -> '.join(paths[v])}")

    # Example 2: Detecting negative cycle
    print("\n" + "="*50)
    print("Example 2: Graph with negative cycle")
    edges2 = [
        ('A', 'B', 1),
        ('B', 'C', -3),
        ('C', 'A', 1)  # Creates negative cycle: A -> B -> C -> A (total: -1)
    ]
    vertices2 = ['A', 'B', 'C']

    has_cycle, cycle = detect_negative_cycle(edges2, vertices2)
    print(f"Negative cycle detected: {has_cycle}")
    if cycle:
        print(f"Cycle: {' -> '.join(cycle)}")

    try:
        bellman_ford(edges2, vertices2, 'A')
    except ValueError as e:
        print(f"\nBellman-Ford correctly raised error: {e}")

    # Example 3: Comparison with Dijkstra's limitation
    print("\n" + "="*50)
    print("Example 3: Why we need Bellman-Ford")
    print("(Dijkstra fails with negative weights, Bellman-Ford succeeds)")

    edges3 = [
        ('S', 'A', 10),
        ('S', 'B', 5),
        ('B', 'A', -7),  # Negative edge makes direct path non-optimal
        ('A', 'T', 1),
        ('B', 'T', 2)
    ]
    vertices3 = ['S', 'A', 'B', 'T']

    distances, paths = bellman_ford(edges3, vertices3, 'S')
    print(f"\nShortest path from S to T: {' -> '.join(paths['T'])}")
    print(f"Distance: {distances['T']}")
    print("(Goes through B despite longer initial distance, due to negative edge)")
