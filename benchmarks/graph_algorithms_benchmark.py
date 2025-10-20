"""
Benchmark script for graph algorithms.

This script measures and compares the performance of different graph algorithms
(Dijkstra, A*, Bellman-Ford) across various graph sizes.

Usage:
    python benchmarks/graph_algorithms_benchmark.py
"""

import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.graphs.dijkstra import dijkstra, dijkstra_single_target
from algorithms.graphs.astar import astar, manhattan_distance, zero_heuristic
from algorithms.graphs.bellman_ford import bellman_ford


def generate_random_graph(
    num_vertices: int,
    edge_probability: float = 0.3,
    max_weight: int = 100
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Generate a random graph for benchmarking.

    Args:
        num_vertices: Number of vertices in the graph
        edge_probability: Probability of edge existing between any two vertices
        max_weight: Maximum edge weight

    Returns:
        Adjacency list representation of graph
    """
    vertices = [f"V{i}" for i in range(num_vertices)]
    graph = {v: [] for v in vertices}

    for i, v1 in enumerate(vertices):
        for v2 in vertices[i+1:]:
            if random.random() < edge_probability:
                weight = random.randint(1, max_weight)
                graph[v1].append((v2, weight))
                graph[v2].append((v1, weight))  # Undirected graph

    return graph


def graph_to_edge_list(
    graph: Dict[str, List[Tuple[str, float]]]
) -> Tuple[List[Tuple[str, str, float]], List[str]]:
    """Convert adjacency list to edge list for Bellman-Ford."""
    edges = []
    seen = set()

    for node, neighbors in graph.items():
        for neighbor, weight in neighbors:
            edge = tuple(sorted([node, neighbor]))
            if edge not in seen:
                edges.append((node, neighbor, weight))
                seen.add(edge)

    vertices = list(graph.keys())
    return edges, vertices


def benchmark_algorithm(
    name: str,
    func: Callable,
    *args,
    **kwargs
) -> Tuple[float, bool]:
    """
    Benchmark an algorithm.

    Args:
        name: Name of the algorithm
        func: Function to benchmark
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function

    Returns:
        Tuple of (execution_time_ms, success)
    """
    try:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        execution_time = (end_time - start_time) * 1000  # Convert to ms
        return execution_time, True

    except Exception as e:
        print(f"    Error in {name}: {e}")
        return 0.0, False


def run_benchmarks() -> Dict:
    """Run comprehensive benchmarks on graph algorithms."""
    print("=" * 70)
    print("GRAPH ALGORITHMS PERFORMANCE BENCHMARK")
    print("=" * 70)

    test_sizes = [10, 50, 100, 500, 1000]
    results = {}

    for size in test_sizes:
        print(f"\n{'='*70}")
        print(f"Testing with {size} vertices")
        print(f"{'='*70}")

        # Generate test graph
        graph = generate_random_graph(size, edge_probability=0.15)

        # Count edges
        num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2

        print(f"Generated graph: {size} vertices, {num_edges} edges")
        print(f"Average degree: {num_edges * 2 / size:.2f}")

        start_vertex = list(graph.keys())[0]
        target_vertex = list(graph.keys())[-1]

        results[size] = {
            'vertices': size,
            'edges': num_edges,
            'algorithms': {}
        }

        # Benchmark Dijkstra (all paths)
        print(f"\n  Dijkstra (all shortest paths from {start_vertex}):")
        time_taken, success = benchmark_algorithm(
            "Dijkstra",
            dijkstra,
            graph,
            start_vertex
        )
        if success:
            print(f"    Time: {time_taken:.2f} ms")
            results[size]['algorithms']['dijkstra_all'] = time_taken

        # Benchmark Dijkstra (single target)
        print(f"\n  Dijkstra (single target {start_vertex} -> {target_vertex}):")
        time_taken, success = benchmark_algorithm(
            "Dijkstra (single)",
            dijkstra_single_target,
            graph,
            start_vertex,
            target_vertex
        )
        if success:
            print(f"    Time: {time_taken:.2f} ms")
            results[size]['algorithms']['dijkstra_single'] = time_taken

        # Benchmark A* with zero heuristic (equivalent to Dijkstra)
        print(f"\n  A* with zero heuristic ({start_vertex} -> {target_vertex}):")
        time_taken, success = benchmark_algorithm(
            "A* (zero)",
            astar,
            graph,
            start_vertex,
            target_vertex,
            zero_heuristic
        )
        if success:
            print(f"    Time: {time_taken:.2f} ms")
            results[size]['algorithms']['astar_zero'] = time_taken

        # Benchmark Bellman-Ford (only for smaller graphs due to O(VE) complexity)
        if size <= 500:
            edges, vertices = graph_to_edge_list(graph)
            print(f"\n  Bellman-Ford (all shortest paths from {start_vertex}):")
            time_taken, success = benchmark_algorithm(
                "Bellman-Ford",
                bellman_ford,
                edges,
                vertices,
                start_vertex
            )
            if success:
                print(f"    Time: {time_taken:.2f} ms")
                results[size]['algorithms']['bellman_ford'] = time_taken
        else:
            print(f"\n  Bellman-Ford: Skipped (too slow for {size} vertices)")

        print()

    return results


def print_summary(results: Dict) -> None:
    """Print benchmark summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n{:<12} {:<10} {:<15} {:<15} {:<15}".format(
        "Vertices", "Edges", "Dijkstra (ms)", "A* (ms)", "Bellman-Ford (ms)"
    ))
    print("-" * 70)

    for size, data in sorted(results.items()):
        algos = data['algorithms']
        dijkstra_time = algos.get('dijkstra_all', 0)
        astar_time = algos.get('astar_zero', 0)
        bellman_time = algos.get('bellman_ford', 0)

        bellman_str = f"{bellman_time:.2f}" if bellman_time > 0 else "N/A"

        print("{:<12} {:<10} {:<15.2f} {:<15.2f} {:<15}".format(
            size, data['edges'], dijkstra_time, astar_time, bellman_str
        ))

    print("\nObservations:")
    print("- Dijkstra: O((V+E) log V) - Fast for most practical graphs")
    print("- A* (zero heuristic): Similar to Dijkstra, slight overhead")
    print("- Bellman-Ford: O(VE) - Slower but handles negative weights")


def save_results(results: Dict, filename: str = "graph_benchmark_results.json") -> None:
    """Save benchmark results to JSON file."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    print("\nStarting graph algorithms benchmark...")
    print("This may take a minute for larger graphs.\n")

    # Seed for reproducibility
    random.seed(42)

    try:
        results = run_benchmarks()
        print_summary(results)
        save_results(results)

        print("\n" + "=" * 70)
        print("Benchmark complete!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
