# Step-by-Step Explanation: KruskalMain.c

Let’s break down the code **line by line** and **section by section**, explaining everything in detail. I’ll use simple language, examples, and diagrams to make it as clear as possible.

---

### **1. Header Files**
```c
#include <stdio.h>
#include "ALGraphKruskal.h"
```

#### **What it does:**
- These lines include necessary libraries and header files for the program to work.
- `#include <stdio.h>` includes the standard input/output library, which provides functions like `printf` for printing to the console.
- `#include "ALGraphKruskal.h"` includes a custom header file that defines the graph data structure and functions used in the program.

#### **Why it’s used:**
- `stdio.h` is needed for printing results to the console.
- `ALGraphKruskal.h` contains the definitions for the graph structure (`ALGraph`) and functions like `GraphInit`, `AddEdge`, `ConKruskalMST`, etc. Without this, the program wouldn’t know what these functions or structures are.

---

### **2. Main Function**
```c
int main(void)
{
    ALGraph graph;
    GraphInit(&graph, 6);      // A, B, C, D, E, F, G
```

#### **What it does:**
- The `main` function is the entry point of the program. Everything inside it is executed when the program runs.
- `ALGraph graph;` declares a variable `graph` of type `ALGraph`. This is the graph data structure that will store the vertices and edges.
- `GraphInit(&graph, 6);` initializes the graph with 6 vertices. The `&graph` passes the address of the graph to the function so it can modify the graph directly.

#### **Why it’s used:**
- The `ALGraph` structure is used to represent the graph in memory. It likely contains:
  - A list of vertices (A, B, C, D, E, F).
  - A list of edges connecting these vertices.
- `GraphInit` initializes the graph by setting up the necessary memory and data structures to store the vertices and edges.

#### **Example:**
Imagine the graph as a blank canvas. `GraphInit` sets up the canvas by drawing 6 empty circles (vertices) labeled A, B, C, D, E, and F. These circles are ready to be connected with lines (edges).

---

### **3. Adding Edges**
```c
    AddEdge(&graph, A, B, 9);
    AddEdge(&graph, B, C, 2);
    AddEdge(&graph, A, C, 12);
    AddEdge(&graph, A, D, 8);
    AddEdge(&graph, D, C, 6);
    AddEdge(&graph, A, F, 11);
    AddEdge(&graph, F, D, 4);
    AddEdge(&graph, D, E, 3);
    AddEdge(&graph, E, C, 7);
    AddEdge(&graph, F, E, 13);
```

#### **What it does:**
- These lines add edges to the graph. Each `AddEdge` call connects two vertices with a specific weight.
- For example, `AddEdge(&graph, A, B, 9);` connects vertex A to vertex B with an edge of weight 9.

#### **Why it’s used:**
- Edges define the connections between vertices. The weights represent the cost or distance between vertices.
- This builds the graph structure so that Kruskal’s algorithm can process it.

#### **Example:**
Think of the vertices as cities and the edges as roads connecting them. The weight is the distance between cities. For example:
- A road connects city A to city B, and it’s 9 miles long.
- Another road connects city B to city C, and it’s 2 miles long.

---

### **4. Displaying Graph Information**
```c
    ShowGraphEdgeInfo(&graph);
    ShowGraphEdgeWeightInfo(&graph);
```

#### **What it does:**
- `ShowGraphEdgeInfo(&graph);` displays the edges of the graph (which vertices are connected).
- `ShowGraphEdgeWeightInfo(&graph);` displays the weights of the edges.

#### **Why it’s used:**
- These functions help visualize the graph before applying Kruskal’s algorithm. This is useful for debugging and understanding the graph structure.

#### **Example:**
If the graph has edges A-B (9), B-C (2), and A-C (12), the output might look like:
```
Edges:
A -> B
B -> C
A -> C

Weights:
A-B: 9
B-C: 2
A-C: 12
```

---

### **5. Applying Kruskal’s Algorithm**
```c
    ConKruskalMST(&graph);
    printf("\n\nPerformed ConKruskal Minimum Spanning Tree \n\n");
```

#### **What it does:**
- `ConKruskalMST(&graph);` applies Kruskal’s algorithm to find the Minimum Spanning Tree (MST) of the graph.
- The `printf` statement simply prints a message to indicate that the algorithm has been applied.

#### **Why it’s used:**
- Kruskal’s algorithm finds the subset of edges that connects all vertices with the minimum total weight, without forming any cycles.

#### **How it works:**
1. **Sort all edges by weight**: The algorithm starts by sorting the edges in ascending order of their weights.
2. **Add edges to the MST**: It iterates through the sorted edges, adding each edge to the MST if it doesn’t form a cycle.
3. **Cycle detection**: To detect cycles, it uses a **Disjoint Set Union (DSU)** data structure (also called Union-Find). This keeps track of which vertices are connected.

#### **Example:**
Imagine the edges are sorted as:
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

The algorithm adds edges in this order, skipping any that would create a cycle. For example:
- Add B-C (2): No cycle.
- Add D-E (3): No cycle.
- Add F-D (4): No cycle.
- Add D-C (6): No cycle.
- Add E-C (7): This would create a cycle (C-D-E-C), so it’s skipped.
- Continue until all vertices are connected.

---

### **6. Displaying the MST**
```c
    ShowGraphEdgeInfo(&graph);
    ShowGraphEdgeWeightInfo(&graph);
```

#### **What it does:**
- These functions display the edges and weights of the MST after Kruskal’s algorithm has been applied.

#### **Why it’s used:**
- This shows the final result of the algorithm: the subset of edges that form the MST.

#### **Example:**
The output might look like:
```
Edges in MST:
B -> C
D -> E
F -> D
D -> C

Weights in MST:
B-C: 2
D-E: 3
F-D: 4
D-C: 6
```

---

### **7. Cleaning Up**
```c
    GraphDestroy(&graph);
    return 0;
}
```

#### **What it does:**
- `GraphDestroy(&graph);` deallocates the memory used by the graph.
- `return 0;` indicates that the program has finished successfully.

#### **Why it’s used:**
- `GraphDestroy` ensures that all dynamically allocated memory is freed, preventing memory leaks.
- `return 0;` is a standard way to indicate that the program has completed without errors.

---

### **Summary**
This code:
1. Initializes a graph with 6 vertices.
2. Adds edges with specific weights.
3. Displays the graph’s edges and weights.
4. Applies Kruskal’s algorithm to find the MST.
5. Displays the edges and weights of the MST.
6. Cleans up memory.

Each step is carefully designed to build, process, and visualize the graph, making it a complete implementation of Kruskal’s algorithm.