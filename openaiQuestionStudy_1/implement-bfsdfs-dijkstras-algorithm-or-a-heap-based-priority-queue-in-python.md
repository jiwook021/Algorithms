# Implement BFS/DFS, Dijkstra's algorithm, or a heap-based priority queue in Python.

Certainly! Let's implement Dijkstra's algorithm, which is used to find the shortest paths from a source node to all other nodes in a graph. This algorithm can be effectively implemented using a priority queue. In Python, the `heapq` module can be used to implement the priority queue.

Here is a step-by-step implementation of Dijkstra's algorithm in Python:

```python
import heapq

def dijkstra(graph, start):
    # The graph is represented as a dictionary of dictionaries
    # Example: graph = {'A': {'B': 1, 'C': 4}, 'B': {'A': 1, 'C': 2, 'D': 5}, ...}
    
    # Priority queue (min-heap), starts with the distance from start node to itself (0)
    priority_queue = [(0, start)]
    
    # To keep track of the minimum distance to each node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    # To keep track of the path that leads to the minimum distance
    previous_nodes = {node: None for node in graph}
    
    while priority_queue:
        # Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Nodes can get added to the priority queue multiple times. We only
        # process a vertex the first time we remove it from the priority queue.
        if current_distance > distances[current_node]:
            continue
        
        # Visit each neighbor of the current node
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            # Only consider this new path if it's better
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances, previous_nodes

# Example usage:
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

start_node = 'A'
distances, previous_nodes = dijkstra(graph, start_node)

print("Shortest Distances:", distances)
print("Previous Nodes:", previous_nodes)
```

### Explanation:
1. **Graph Representation**: The graph is represented as a dictionary of dictionaries. For each key (node), there is a corresponding dictionary that holds the neighbors of the node and the weight of the edge to each neighbor.

2. **Priority Queue**: We use a priority queue (min-heap) to always extend the path from the node having the smallest known distance.

3. **Distance and Previous Nodes Tracking**:
   - `distances`: Dictionary that holds the shortest known distance from the start node to every other node.
   - `previous_nodes`: Dictionary that helps to reconstruct the shortest path from start node to any node.

4. **Algorithm Loop**:
   - Extract the node with the smallest distance from the priority queue.
   - Update the distances to its neighboring nodes if a shorter path is found.

5. **Output**:
   - `distances`: Contains the shortest distances from the start node to all other nodes.
   - `previous_nodes`: Can be used to reconstruct the shortest paths.

This implementation provides a clear and efficient way to execute Dijkstra's algorithm using Python's `heapq` for the priority queue.