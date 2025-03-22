# Step-by-Step Explanation: ALGraphKruskal.c

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. I’ll also define technical terms and explain the reasoning behind the code’s design.

---

### **1. Graph Initialization (`GraphInit`)**
```c
void GraphInit(ALGraph * pg, int nv)
{
    int i;	

    pg->adjList = (List*)malloc(sizeof(List)*nv);
    pg->numV = nv;
    pg->numE = 0;

    for(i=0; i<nv; i++)
    {
        ListInit(&(pg->adjList[i]));
        SetSortRule(&(pg->adjList[i]), WhoIsPrecede); 
    }

    pg->visitInfo= (int *)malloc(sizeof(int) * pg->numV);
    memset(pg->visitInfo, 0, sizeof(int) * pg->numV);

    PQueueInit(&(pg->pqueue), PQWeightComp);
}
```

#### **What It Does**
This function initializes the graph. It sets up the adjacency list, visit information array, and priority queue.

#### **Step-by-Step Breakdown**
1. **Allocate Memory for Adjacency List**:
   - `pg->adjList = (List*)malloc(sizeof(List)*nv);`
   - Allocates memory for an array of linked lists (one for each vertex).
   - Each linked list will store the adjacent vertices for a given vertex.

2. **Set Number of Vertices and Edges**:
   - `pg->numV = nv;` sets the number of vertices.
   - `pg->numE = 0;` initializes the number of edges to 0.

3. **Initialize Each Adjacency List**:
   - The `for` loop iterates over each vertex:
     - `ListInit(&(pg->adjList[i]));` initializes the linked list for vertex `i`.
     - `SetSortRule(&(pg->adjList[i]), WhoIsPrecede);` sets a sorting rule for the linked list (explained later).

4. **Allocate Memory for Visit Information**:
   - `pg->visitInfo= (int *)malloc(sizeof(int) * pg->numV);`
   - Allocates memory for an array to track visited vertices during traversal.
   - `memset(pg->visitInfo, 0, sizeof(int) * pg->numV);` initializes all elements to 0 (unvisited).

5. **Initialize Priority Queue**:
   - `PQueueInit(&(pg->pqueue), PQWeightComp);`
   - Initializes the priority queue with a comparison function (`PQWeightComp`) to sort edges by weight.

#### **Why This Approach?**
- **Adjacency List**: Efficient for sparse graphs (few edges relative to vertices).
- **Priority Queue**: Essential for Kruskal’s algorithm to process edges in ascending order of weight.
- **Visit Information**: Used for cycle detection during graph traversal.

---

### **2. Adding an Edge (`AddEdge`)**
```c
void AddEdge(ALGraph * pg, int fromV, int toV, int weight)
{
    Edge edge = {fromV, toV, weight};

    LInsert(&(pg->adjList[fromV]), toV);
    LInsert(&(pg->adjList[toV]), fromV);
    pg->numE += 1;

    PEnqueue(&(pg->pqueue), edge);
}
```

#### **What It Does**
Adds an undirected edge between two vertices and stores it in the priority queue.

#### **Step-by-Step Breakdown**
1. **Create Edge**:
   - `Edge edge = {fromV, toV, weight};`
   - Defines an edge with its source (`fromV`), destination (`toV`), and weight.

2. **Insert into Adjacency Lists**:
   - `LInsert(&(pg->adjList[fromV]), toV);`
   - Adds `toV` to the adjacency list of `fromV`.
   - `LInsert(&(pg->adjList[toV]), fromV);`
   - Adds `fromV` to the adjacency list of `toV` (since the graph is undirected).

3. **Increment Edge Count**:
   - `pg->numE += 1;`
   - Increases the total number of edges in the graph.

4. **Enqueue Edge**:
   - `PEnqueue(&(pg->pqueue), edge);`
   - Adds the edge to the priority queue, sorted by weight.

#### **Why This Approach?**
- **Undirected Graph**: Both vertices are added to each other’s adjacency lists.
- **Priority Queue**: Ensures edges are processed in ascending order of weight for Kruskal’s algorithm.

---

### **3. Kruskal’s Algorithm (`ConKruskalMST`)**
```c
void ConKruskalMST(ALGraph * pg)
{
    Edge recvEdge[20];
    Edge edge;
    int eidx = 0;
    int i;

    while(pg-> numE+1 > pg->numV)
    {
        edge = PDequeue(&(pg->pqueue));
        RemoveEdge(pg, edge.v1, edge.v2);

        if(!IsConnVertex(pg, edge.v1, edge.v2))
        {
           RecoverEdge(pg, edge.v1, edge.v2, edge.weight);
           recvEdge[eidx++] = edge;
        }
    }

    for(i = 0; i<eidx ;i++)
        PEnqueue(&(pg->pqueue),recvEdge[i]);
}
```

#### **What It Does**
Constructs the Minimum Spanning Tree (MST) using Kruskal’s algorithm.

#### **Step-by-Step Breakdown**
1. **Initialize Variables**:
   - `Edge recvEdge[20];` stores edges that are part of the MST.
   - `Edge edge;` temporarily holds the current edge being processed.
   - `int eidx = 0;` tracks the number of edges in the MST.

2. **Main Loop**:
   - `while(pg-> numE+1 > pg->numV)`
   - Continues until the number of edges in the MST is one less than the number of vertices (a tree property).

3. **Dequeue Edge**:
   - `edge = PDequeue(&(pg->pqueue));`
   - Removes the smallest edge from the priority queue.

4. **Remove Edge from Graph**:
   - `RemoveEdge(pg, edge.v1, edge.v2);`
   - Temporarily removes the edge from the graph.

5. **Check for Cycle**:
   - `if(!IsConnVertex(pg, edge.v1, edge.v2))`
   - Uses DFS to check if removing the edge disconnects the graph (i.e., the edge doesn’t form a cycle).

6. **Recover Edge**:
   - If no cycle is formed, the edge is added back to the graph and stored in `recvEdge`.

7. **Re-enqueue Edges**:
   - After the loop, edges in `recvEdge` are re-enqueued into the priority queue.

#### **Why This Approach?**
- **Greedy Algorithm**: Kruskal’s algorithm selects the smallest edge at each step, ensuring the MST has the minimum total weight.
- **Cycle Detection**: Ensures the MST remains a tree (no cycles).

---

### **4. Cycle Detection (`IsConnVertex`)**
```c
int IsConnVertex(ALGraph * pg, int v1, int v2)
{
    Stack stack;
    int visitV = v1;
    int nextV;

    StackInit(&stack);
    VisitVertex(pg, visitV);
    SPush(&stack, visitV);

    while(LFirst(&(pg->adjList[visitV]), &nextV) == TRUE)
    {
        int visitFlag = FALSE;

        if(nextV == v2)
        {
            memset(pg->visitInfo, 0, sizeof(int)* pg->numV);
            return TRUE;
        }

        if(VisitVertex(pg,nextV)==TRUE)
        {
            SPush(&stack, visitV);
            visitV = nextV;
            visitFlag = TRUE;
        }
        else
        {
            while(LNext(&(pg->adjList[visitV]), &nextV) == TRUE)
            {
                if(nextV == v2)
                {
                    memset(pg->visitInfo, 0, sizeof(int) * pg->numV);
                    return TRUE;
                }
                if(VisitVertex(pg,nextV) == TRUE){
                    SPush(&stack, visitV);
                    visitV = nextV;
                    visitFlag = TRUE;
                    break;
                }
            }
        }
        if(visitFlag == FALSE)
        {
            if(SIsEmpty(&stack) == TRUE)
               break;
            else
                visitV = SPop(&stack);
        }
    }
    memset(pg->visitInfo, 0, sizeof(int) *pg->numV);
    return FALSE;
}
```

#### **What It Does**
Checks if two vertices are connected using DFS.

#### **Step-by-Step Breakdown**
1. **Initialize Stack**:
   - `Stack stack;` is used to track the traversal path.
   - `StackInit(&stack);` initializes the stack.

2. **Start DFS**:
   - `visitV = v1;` starts traversal from `v1`.
   - `VisitVertex(pg, visitV);` marks `v1` as visited.
   - `SPush(&stack, visitV);` pushes `v1` onto the stack.

3. **Traverse Adjacency List**:
   - `while(LFirst(&(pg->adjList[visitV]), &nextV) == TRUE)`
   - Iterates through the adjacency list of `visitV`.

4. **Check for Target Vertex**:
   - If `nextV == v2`, the vertices are connected.

5. **Visit Next Vertex**:
   - If `nextV` is unvisited, it is marked as visited and pushed onto the stack.

6. **Backtrack**:
   - If no unvisited vertices are found, the algorithm backtracks using the stack.

7. **Reset Visit Information**:
   - `memset(pg->visitInfo, 0, sizeof(int) *pg->numV);`
   - Resets the visit information array after traversal.

#### **Why This Approach?**
- **DFS**: Efficiently explores all paths between two vertices.
- **Stack**: Tracks the traversal path for backtracking.

---

This explanation covers the core parts of the code. Let me know if you’d like further clarification or a deeper dive into specific sections!