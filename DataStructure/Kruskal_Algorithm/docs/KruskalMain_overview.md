# Code Overview: KruskalMain.c

This code is an implementation of **Kruskal's algorithm** for finding the **Minimum Spanning Tree (MST)** of a weighted, undirected graph. Let's break this down step by step, explaining the purpose, the problem being solved, and how the code works.

---

### **Purpose of the Code**
The purpose of this code is to:
1. **Model a graph**: Represent a graph using an adjacency list (ALGraph).
2. **Find the Minimum Spanning Tree (MST)**: Use Kruskal's algorithm to find the MST of the graph.
3. **Display graph information**: Show the edges and their weights before and after applying Kruskal's algorithm.
4. **Clean up resources**: Properly deallocate memory used by the graph.

The **Minimum Spanning Tree (MST)** is a subset of the edges of a connected, undirected graph that connects all the vertices together without any cycles and with the minimum possible total edge weight. Kruskal's algorithm is one of the most efficient ways to solve this problem.

---

### **Problem Being Solved**
The problem being solved is:
- Given a weighted, undirected graph, find the subset of edges that connects all the vertices with the minimum total weight, ensuring no cycles are formed.

This is a classic problem in graph theory with applications in network design, clustering, and optimization.

---

### **Approach Taken**
The code uses the following approach:
1. **Graph Initialization**: The graph is initialized with 6 vertices (labeled A, B, C, D, E, F).
2. **Edge Addition**: Edges are added to the graph, each with an associated weight.
3. **Graph Visualization**: The graph's edges and their weights are displayed.
4. **Kruskal's Algorithm**: The algorithm is applied to find the MST.
5. **Result Visualization**: The edges and weights of the MST are displayed.
6. **Cleanup**: The graph's memory is deallocated.

---

### **Algorithms Used**
1. **Kruskal's Algorithm**:
   - This algorithm works by sorting all the edges in the graph in ascending order of their weights.
   - It then iterates through the sorted edges, adding them to the MST if they do not form a cycle.
   - To detect cycles, it uses a **Disjoint Set Union (DSU)** data structure (also known as Union-Find).

2. **Graph Representation**:
   - The graph is represented using an **Adjacency List (ALGraph)**, which is a common way to store graphs in memory. Each vertex has a list of its neighboring vertices and the weights of the connecting edges.

---

### **Overall Structure**
The code is structured as follows:
1. **Graph Initialization**:
   - `GraphInit(&graph, 6)` initializes the graph with 6 vertices.
2. **Edge Addition**:
   - `AddEdge(&graph, ...)` adds edges between vertices with specific weights.
3. **Graph Visualization**:
   - `ShowGraphEdgeInfo(&graph)` displays the edges of the graph.
   - `ShowGraphEdgeWeightInfo(&graph)` displays the weights of the edges.
4. **Kruskal's Algorithm**:
   - `ConKruskalMST(&graph)` applies Kruskal's algorithm to find the MST.
5. **Result Visualization**:
   - The edges and weights of the MST are displayed again.
6. **Cleanup**:
   - `GraphDestroy(&graph)` deallocates memory used by the graph.

---

### **How the Parts Work Together**
1. **Graph Initialization**:
   - The graph is initialized with 6 vertices, labeled A, B, C, D, E, and F.
2. **Edge Addition**:
   - Edges are added to the graph, defining the connections between vertices and their weights.
3. **Graph Visualization**:
   - Before applying Kruskal's algorithm, the graph's edges and weights are displayed to show the initial state.
4. **Kruskal's Algorithm**:
   - The algorithm processes the edges, sorts them by weight, and constructs the MST by adding edges that do not form cycles.
5. **Result Visualization**:
   - After applying Kruskal's algorithm, the edges and weights of the MST are displayed to show the final result.
6. **Cleanup**:
   - The graph's memory is deallocated to prevent memory leaks.

---

### **Key Concepts**
1. **Graph Representation**:
   - The graph is represented using an adjacency list, which is efficient for sparse graphs (graphs with relatively few edges compared to vertices).
2. **Kruskal's Algorithm**:
   - The algorithm relies on sorting edges and using the Union-Find data structure to detect cycles.
3. **Minimum Spanning Tree (MST)**:
   - The MST is a tree that spans all vertices with the minimum total edge weight.

---

### **Example Walkthrough**
Let's walk through the example graph defined in the code:
- Vertices: A, B, C, D, E, F
- Edges and weights:
  - A-B: 9
  - B-C: 2
  - A-C: 12
  - A-D: 8
  - D-C: 6
  - A-F: 11
  - F-D: 4
  - D-E: 3
  - E-C: 7
  - F-E: 13

Kruskal's algorithm will:
1. Sort the edges by weight:
   - B-C: 2
   - D-E: 3
   - F-D: 4
   - D-C: 6
   - E-C: 7
   - A-D: 8
   - A-B: 9
   - A-F: 11
   - A-C: 12
   - F-E: 13
2. Add edges to the MST in order, skipping any that form a cycle.

The final MST will include the edges with the smallest weights that connect all vertices without cycles.

---

### **Summary**
This code demonstrates how to:
1. Represent a graph using an adjacency list.
2. Apply Kruskal's algorithm to find the MST.
3. Visualize the graph and its MST.
4. Clean up resources after use.

It is a complete implementation of Kruskal's algorithm, showcasing both the algorithm and good programming practices like memory management.