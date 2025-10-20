"""
Dijkstra's Shortest Path Algorithm.

This module implements Dijkstra's algorithm for finding the shortest path
from a source vertex to all other vertices in a weighted graph with
non-negative edge weights.

Time Complexity: O((V + E) log V) with binary heap
Space Complexity: O(V)

Example:
    >>> graph = {
    ...     'A': [('B', 4), ('C', 2)],
    ...     'B': [('C', 1), ('D', 5)],
    ...     'C': [('D', 8), ('E', 10)],
    ...     'D': [('E', 2)],
    ...     'E': []
    ... }
    >>> distances, paths = dijkstra(graph, 'A')
    >>> distances['E']
    11
    >>> paths['E']
    ['A', 'B', 'C', 'D', 'E']
"""

import heapq
from typing import Dict, List, Tuple, Optional


def dijkstra(
    graph: Dict[str, List[Tuple[str, float]]],
    start: str
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Find shortest paths from start vertex to all other vertices using Dijkstra's algorithm.

    Args:
        graph: Adjacency list representation where graph[node] = [(neighbor, weight), ...]
        start: Starting vertex

    Returns:
        A tuple containing:
        - distances: Dictionary mapping each vertex to its shortest distance from start
        - paths: Dictionary mapping each vertex to the shortest path from start

    Raises:
        ValueError: If start vertex is not in the graph
        ValueError: If graph contains negative edge weights

    Time Complexity: O((V + E) log V) where V is vertices and E is edges
    Space Complexity: O(V) for storing distances and paths
    """
    if start not in graph:
        raise ValueError(f"Start vertex '{start}' not found in graph")

    # Validate non-negative weights
    for node in graph:
        for neighbor, weight in graph[node]:
            if weight < 0:
                raise ValueError(
                    f"Negative edge weight found: {node} -> {neighbor} = {weight}. "
                    "Use Bellman-Ford algorithm for graphs with negative weights."
                )

    # Initialize distances with infinity for all nodes except start
    distances: Dict[str, float] = {node: float('inf') for node in graph}
    distances[start] = 0

    # Initialize paths
    paths: Dict[str, List[str]] = {node: [] for node in graph}
    paths[start] = [start]

    # Priority queue: (distance, node)
    # Using heap to efficiently get the node with minimum distance
    pq: List[Tuple[float, str]] = [(0, start)]

    # Keep track of visited nodes to avoid reprocessing
    visited: set = set()

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # Skip if we've already processed this node
        if current_node in visited:
            continue

        visited.add(current_node)

        # Skip if this is an outdated entry in the priority queue
        if current_distance > distances[current_node]:
            continue

        # Explore neighbors
        for neighbor, weight in graph[current_node]:
            # Calculate new distance through current node
            new_distance = current_distance + weight

            # If we found a shorter path, update it
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                paths[neighbor] = paths[current_node] + [neighbor]
                heapq.heappush(pq, (new_distance, neighbor))

    return distances, paths


def dijkstra_single_target(
    graph: Dict[str, List[Tuple[str, float]]],
    start: str,
    target: str
) -> Tuple[float, List[str]]:
    """
    Find shortest path from start to target vertex (optimized version).

    This is an optimized version that terminates early once the target is reached.

    Args:
        graph: Adjacency list representation where graph[node] = [(neighbor, weight), ...]
        start: Starting vertex
        target: Target vertex

    Returns:
        A tuple containing:
        - distance: Shortest distance from start to target
        - path: Shortest path from start to target

    Raises:
        ValueError: If start or target vertex is not in the graph
        ValueError: If no path exists from start to target

    Time Complexity: O((V + E) log V) worst case, often better with early termination
    Space Complexity: O(V)
    """
    if start not in graph:
        raise ValueError(f"Start vertex '{start}' not found in graph")
    if target not in graph:
        raise ValueError(f"Target vertex '{target}' not found in graph")

    distances: Dict[str, float] = {node: float('inf') for node in graph}
    distances[start] = 0

    paths: Dict[str, List[str]] = {node: [] for node in graph}
    paths[start] = [start]

    pq: List[Tuple[float, str]] = [(0, start)]
    visited: set = set()

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # Early termination: found shortest path to target
        if current_node == target:
            return distances[target], paths[target]

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            new_distance = current_distance + weight

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                paths[neighbor] = paths[current_node] + [neighbor]
                heapq.heappush(pq, (new_distance, neighbor))

    # If we exit the loop without finding target, no path exists
    raise ValueError(f"No path exists from '{start}' to '{target}'")


if __name__ == "__main__":
    # Example usage and demonstration
    sample_graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('C', 1), ('D', 5)],
        'C': [('D', 8), ('E', 10)],
        'D': [('E', 2), ('F', 6)],
        'E': [('F', 3)],
        'F': []
    }

    print("Graph:")
    for node, edges in sample_graph.items():
        print(f"  {node}: {edges}")

    print("\nRunning Dijkstra's algorithm from vertex 'A':")
    distances, paths = dijkstra(sample_graph, 'A')

    print("\nShortest distances from A:")
    for node, distance in sorted(distances.items()):
        print(f"  A -> {node}: {distance}")

    print("\nShortest paths from A:")
    for node, path in sorted(paths.items()):
        print(f"  {' -> '.join(path)}")

    print("\nFinding path from A to F:")
    dist, path = dijkstra_single_target(sample_graph, 'A', 'F')
    print(f"  Distance: {dist}")
    print(f"  Path: {' -> '.join(path)}")
