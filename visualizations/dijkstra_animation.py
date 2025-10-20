"""
Visualization of Dijkstra's Algorithm.

This script creates an animated visualization showing how Dijkstra's algorithm
explores nodes and finds shortest paths in a graph.

Requirements:
    - matplotlib
    - networkx (optional, for better graph layouts)

Usage:
    python visualizations/dijkstra_animation.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import algorithms
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Set
import heapq


def visualize_dijkstra_step_by_step(
    graph: Dict[str, List[Tuple[str, float]]],
    start: str,
    positions: Dict[str, Tuple[float, float]]
) -> None:
    """
    Visualize Dijkstra's algorithm step by step.

    Args:
        graph: Adjacency list representation
        start: Starting vertex
        positions: Dictionary mapping vertices to (x, y) coordinates for plotting
    """
    # Initialize
    distances: Dict[str, float] = {node: float('inf') for node in graph}
    distances[start] = 0
    pq: List[Tuple[float, str]] = [(0, start)]
    visited: Set[str] = set()
    parent: Dict[str, str] = {}

    # Setup the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("Dijkstra's Algorithm Visualization", fontsize=16, fontweight='bold')

    step = 0

    while pq:
        step += 1
        ax.clear()

        current_distance, current_node = heapq.heappop(pq)

        if current_node in visited:
            continue

        visited.add(current_node)

        # Draw edges
        for node, edges in graph.items():
            for neighbor, weight in edges:
                x1, y1 = positions[node]
                x2, y2 = positions[neighbor]

                # Color edge based on state
                if node in parent and parent[node] == neighbor:
                    color = 'green'
                    width = 3
                    alpha = 1.0
                elif neighbor in parent and parent[neighbor] == node:
                    color = 'green'
                    width = 3
                    alpha = 1.0
                elif node == current_node or neighbor == current_node:
                    color = 'orange'
                    width = 2
                    alpha = 0.8
                else:
                    color = 'gray'
                    width = 1
                    alpha = 0.3

                ax.plot([x1, x2], [y1, y2], color=color, linewidth=width,
                       alpha=alpha, zorder=1)

                # Draw edge weight
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y, f'{weight}', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='gray', alpha=0.7))

        # Draw nodes
        for node, (x, y) in positions.items():
            if node == current_node:
                color = 'orange'
                size = 800
                label = f'{node}\n{distances[node]:.1f}'
            elif node in visited:
                color = 'lightgreen'
                size = 600
                label = f'{node}\n{distances[node]:.1f}'
            elif distances[node] != float('inf'):
                color = 'lightyellow'
                size = 600
                label = f'{node}\n{distances[node]:.1f}'
            else:
                color = 'lightgray'
                size = 500
                label = f'{node}\n∞'

            ax.scatter(x, y, s=size, c=color, edgecolors='black',
                      linewidths=2, zorder=2)
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=10, fontweight='bold')

        # Explore neighbors
        for neighbor, weight in graph[current_node]:
            if neighbor not in visited:
                new_distance = current_distance + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parent[neighbor] = current_node
                    heapq.heappush(pq, (new_distance, neighbor))

        # Legend
        legend_elements = [
            mpatches.Patch(color='orange', label='Current Node'),
            mpatches.Patch(color='lightgreen', label='Visited'),
            mpatches.Patch(color='lightyellow', label='In Queue'),
            mpatches.Patch(color='lightgray', label='Unvisited'),
            mpatches.Patch(color='green', label='Shortest Path')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title(f'Step {step}: Processing node {current_node} '
                    f'(distance: {current_distance:.1f})',
                    fontsize=12)
        ax.set_xlim(-1, max(x for x, y in positions.values()) + 1)
        ax.set_ylim(-1, max(y for x, y in positions.values()) + 1)
        ax.axis('off')

        plt.pause(1.0)  # Pause to show each step

    # Final state
    ax.set_title(f'Final: All nodes processed. Shortest paths from {start}',
                fontsize=12, color='green')
    plt.pause(2.0)
    plt.show()


def create_sample_graph() -> Tuple[Dict[str, List[Tuple[str, float]]],
                                   Dict[str, Tuple[float, float]]]:
    """Create a sample graph for visualization."""
    graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('A', 4), ('C', 1), ('D', 5)],
        'C': [('A', 2), ('B', 1), ('D', 8), ('E', 10)],
        'D': [('B', 5), ('C', 8), ('E', 2), ('F', 6)],
        'E': [('C', 10), ('D', 2), ('F', 3)],
        'F': [('D', 6), ('E', 3)]
    }

    # Manually positioned for nice visualization
    positions = {
        'A': (0, 3),
        'B': (2, 4),
        'C': (2, 2),
        'D': (4, 3),
        'E': (4, 1),
        'F': (6, 2)
    }

    return graph, positions


if __name__ == "__main__":
    print("Dijkstra's Algorithm Visualization")
    print("=" * 50)
    print("\nThis will show a step-by-step animated visualization")
    print("of Dijkstra's algorithm finding shortest paths.\n")
    print("Close the window to exit.\n")

    graph, positions = create_sample_graph()

    print("Graph structure:")
    for node, edges in sorted(graph.items()):
        print(f"  {node}: {edges}")

    print(f"\nStarting visualization from node 'A'...")
    print("Watch as the algorithm explores nodes (orange),")
    print("marks them as visited (green), and builds shortest paths!\n")

    try:
        visualize_dijkstra_step_by_step(graph, 'A', positions)
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
    except Exception as e:
        print(f"\n\nError during visualization: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")
