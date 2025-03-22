# Code Overview: main.cpp

### Purpose and Main Functionality of the Code

This C++ code is designed to solve **graph traversal problems**, specifically focusing on finding the **shortest paths** between nodes in a graph. It implements two well-known algorithms for this purpose:

1. **Dijkstra's Algorithm**: Used for finding the shortest paths from a single source node to all other nodes in a graph with **non-negative edge weights**.
2. **Bellman-Ford Algorithm**: Used for finding the shortest paths from a single source node to all other nodes in a graph that may contain **negative edge weights** (though it can handle non-negative weights as well).

The code also hints at a third algorithm, **Floyd-Warshall**, which is used for finding the shortest paths between all pairs of nodes in a graph. However, the implementation of Floyd-Warshall is not fully provided in the code snippet.

### Problem Being Solved

The problem being solved is a classic **shortest path problem** in graph theory. Given a graph with nodes (vertices) and edges (connections between nodes), each edge has an associated weight (cost). The goal is to find the path with the **minimum total weight** from a starting node to all other nodes in the graph.

### Algorithms Used

1. **Dijkstra's Algorithm**:
   - **Purpose**: Finds the shortest paths from a single source node to all other nodes in a graph with non-negative edge weights.
   - **Approach**: Uses a **priority queue** to always expand the least-cost node first, ensuring that the shortest path to each node is found efficiently.
   - **Key Data Structures**:
     - `priority_queue`: A min-heap that prioritizes nodes with the smallest current distance.
     - `dist[]`: An array that stores the shortest distance from the source node to each node.

2. **Bellman-Ford Algorithm**:
   - **Purpose**: Finds the shortest paths from a single source node to all other nodes in a graph that may contain negative edge weights.
   - **Approach**: Iteratively relaxes all edges `N-1` times (where `N` is the number of nodes) to ensure the shortest paths are found. It can also detect negative weight cycles.
   - **Key Data Structures**:
     - `dist[]`: An array that stores the shortest distance from the source node to each node.
     - `a[]`, `b[]`, `w[]`: Arrays that store the edges of the graph (source node, destination node, and weight, respectively).

### Overall Structure of the Code

The code is structured as follows:

1. **Global Variables**:
   - `INF`: A constant representing infinity, used to initialize distances.
   - `N`, `M`: Variables to store the number of nodes and edges in the graph.
   - `dist[]`: An array to store the shortest distances from the source node.
   - `a[]`, `b[]`, `w[]`: Arrays to store the edges of the graph.
   - `adj[]`: An adjacency list to represent the graph.
   - `pq`: A priority queue used in Dijkstra's algorithm.

2. **Dijkstra's Algorithm**:
   - Initializes distances to infinity.
   - Sets the distance of the source node to 0.
   - Uses a priority queue to process nodes in order of increasing distance.
   - Updates distances to neighboring nodes if a shorter path is found.

3. **Bellman-Ford Algorithm**:
   - Initializes distances to infinity.
   - Sets the distance of the source node to 0.
   - Iteratively relaxes all edges `N-1` times to find the shortest paths.
   - Can detect negative weight cycles if further relaxation is possible after `N-1` iterations.

4. **Main Function**:
   - Reads input values for the number of nodes (`N`) and edges (`M`).
   - Reads the edges and constructs the adjacency list.
   - Calls the appropriate algorithm (Dijkstra or Bellman-Ford) to compute shortest paths.

### How the Parts Work Together

- The **main function** reads the graph's structure (number of nodes and edges) and constructs the adjacency list (`adj[]`).
- Depending on the algorithm chosen (Dijkstra or Bellman-Ford), the corresponding function is called to compute the shortest paths from a given source node.
- Both algorithms use the `dist[]` array to store the shortest distances, but they differ in how they process the graph:
  - Dijkstra's algorithm uses a priority queue to efficiently find the next node to process.
  - Bellman-Ford algorithm iteratively relaxes all edges to ensure the shortest paths are found.

### Summary

This code is a versatile tool for solving shortest path problems in graphs. It provides implementations of two fundamental algorithms (Dijkstra and Bellman-Ford) and is structured to handle different types of graphs (with non-negative or negative edge weights). The main function serves as the entry point, reading input and invoking the appropriate algorithm based on the problem's requirements.

In the next questions, we can dive deeper into the **line-by-line explanation** of the code and explore **potential improvements** to make it more robust, efficient, and readable. Let me know how you'd like to proceed!