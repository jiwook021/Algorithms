# Step-by-Step Explanation: main.cpp

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll use simple terms, examples, and diagrams to make everything clear, even for beginners.

---

### **1. Global Variables and Includes**
```cpp
#define INF 999999999
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;
```

#### What It Does:
- `#define INF 999999999`: Defines a constant `INF` (infinity) to represent a very large number. This is used to initialize distances to nodes that haven’t been visited yet.
- `#include <iostream>`: Includes the standard input/output library for reading and writing data.
- `#include <vector>`: Includes the `vector` library, which is used to create dynamic arrays (like `adj[]` for storing the graph).
- `#include <queue>`: Includes the `queue` library, which is used for the priority queue in Dijkstra’s algorithm.
- `#include <algorithm>`: Includes the `algorithm` library, which provides useful functions like sorting (though it’s not used in this code).
- `using namespace std;`: Allows us to use standard library functions (like `cin`, `cout`, `vector`, etc.) without typing `std::` every time.

#### Why It’s Used:
- `INF` is used to represent an unreachable distance. For example, if the distance from node A to node B is `INF`, it means there’s no path between them (yet).
- The libraries are included to provide the necessary tools for working with graphs, queues, and input/output.

---

### **2. Variable Declarations**
```cpp
int N, M, dist[100005], a[100005], b[100005], w[100005], d[1005][1005];
vector<pair<int, int>> adj[100005];
priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
```

#### What It Does:
- `int N, M`: `N` is the number of nodes (vertices) in the graph, and `M` is the number of edges.
- `int dist[100005]`: An array to store the shortest distance from the source node to each node.
- `int a[100005], b[100005], w[100005]`: Arrays to store the edges. For each edge:
  - `a[i]` is the source node.
  - `b[i]` is the destination node.
  - `w[i]` is the weight (cost) of the edge.
- `int d[1005][1005]`: A 2D array that could be used for the Floyd-Warshall algorithm (though it’s not implemented here).
- `vector<pair<int, int>> adj[100005]`: An adjacency list to represent the graph. Each node has a list of pairs, where each pair contains:
  - A neighboring node.
  - The weight of the edge to that node.
- `priority_queue<...> pq`: A priority queue (min-heap) used in Dijkstra’s algorithm to always process the node with the smallest distance first.

#### Why It’s Used:
- `dist[]` is used to store the shortest distances found by the algorithms.
- `a[]`, `b[]`, and `w[]` are used to store the graph’s edges in a simple format.
- `adj[]` is an adjacency list, which is a common way to represent graphs. It’s efficient for sparse graphs (graphs with few edges compared to nodes).
- `pq` is used in Dijkstra’s algorithm to efficiently find the next node to process.

---

### **3. Dijkstra’s Algorithm**
```cpp
void dijkstra(int st){
  for(int i = 1; i <= N; i++){
    dist[i] = INF;
  }
  dist[st] = 0;
  pq.push({ 0, st });
  while(!pq.empty()){
    auto [now_val, now] = pq.top();
    pq.pop();
    if(now_val > dist[now]) continue;
    for(auto [next, next_val] : adj[now]){
      int tot = now_val + next_val;
      if(dist[next] > tot){
        dist[next] = tot;
        pq.push({ tot, next });
      }
    }
  }
}
```

#### What It Does:
1. **Initialization**:
   - Set all distances in `dist[]` to `INF` (infinity), meaning we don’t know the shortest path yet.
   - Set the distance of the starting node (`st`) to `0` because the distance from a node to itself is `0`.
   - Push the starting node into the priority queue with a distance of `0`.

2. **Main Loop**:
   - While the priority queue is not empty:
     - Get the node with the smallest distance (`now`) and its distance (`now_val`).
     - If `now_val` is greater than the current distance in `dist[now]`, skip this node (it’s already processed).
     - For each neighbor (`next`) of `now`:
       - Calculate the total distance (`tot`) to reach `next` through `now`.
       - If this distance is smaller than the current distance in `dist[next]`, update `dist[next]` and push `next` into the priority queue.

#### Why It’s Used:
- Dijkstra’s algorithm is used to find the shortest paths in a graph with **non-negative edge weights**.
- The priority queue ensures that we always process the node with the smallest distance first, which is key to the algorithm’s efficiency.

#### Example:
Imagine a graph with 3 nodes:
- Node 1 is connected to Node 2 with weight `2`.
- Node 1 is connected to Node 3 with weight `4`.
- Node 2 is connected to Node 3 with weight `1`.

Starting from Node 1:
1. Push Node 1 into the queue with distance `0`.
2. Process Node 1:
   - Update Node 2’s distance to `2` and push it into the queue.
   - Update Node 3’s distance to `4` and push it into the queue.
3. Process Node 2:
   - Update Node 3’s distance to `3` (since `2 + 1 < 4`).
4. Process Node 3:
   - No updates needed.

Final distances:
- Node 1: `0`
- Node 2: `2`
- Node 3: `3`

---

### **4. Bellman-Ford Algorithm**
```cpp
void bellman_ford(int st){
  for(int i = 1; i <= N; i++){
    dist[i] = INF;
  }
  dist[st] = 0;

  for(int j = 1; j < N; j++){
    for(int i = 1; i <= M; i++){
      int now = a[i];
      int next = b[i];
      if(dist[now] != INF && dist[next] > dist[now] + w[i]){
        dist[next] = dist[now] + w[i];
      }
    }
  }
}
```

#### What It Does:
1. **Initialization**:
   - Set all distances in `dist[]` to `INF`.
   - Set the distance of the starting node (`st`) to `0`.

2. **Main Loop**:
   - Repeat `N-1` times (where `N` is the number of nodes):
     - For each edge:
       - If the distance to the source node (`now`) is not `INF` and the distance to the destination node (`next`) can be improved, update `dist[next]`.

#### Why It’s Used:
- Bellman-Ford is used to find the shortest paths in a graph that may contain **negative edge weights**.
- It can also detect negative weight cycles (though this part isn’t implemented here).

#### Example:
Imagine a graph with 3 nodes:
- Node 1 is connected to Node 2 with weight `2`.
- Node 2 is connected to Node 3 with weight `-1`.
- Node 1 is connected to Node 3 with weight `4`.

Starting from Node 1:
1. Initialize distances:
   - Node 1: `0`
   - Node 2: `INF`
   - Node 3: `INF`
2. After the first iteration:
   - Node 2: `2`
   - Node 3: `4`
3. After the second iteration:
   - Node 3: `1` (since `2 + (-1) < 4`).

Final distances:
- Node 1: `0`
- Node 2: `2`
- Node 3: `1`

---

### **5. Main Function**
```cpp
int main(){
  cin >> N >> M;
  for(int i = 1; i <= M; i++){
    cin >> a[i] >> b[i] >> w[i];
    adj[a[i]].push_back({ b[i], w[i] });
    adj[b[i]].push_back({ a[i], w[i] });
  }

  // dijkstra(1);
  // bellman_ford(1);
  // floyd_warshall();

  return 0;
}
```

#### What It Does:
1. **Input**:
   - Read the number of nodes (`N`) and edges (`M`).
   - For each edge, read the source node (`a[i]`), destination node (`b[i]`), and weight (`w[i]`).
   - Add the edge to the adjacency list (`adj[]`) for both directions (since the graph is undirected).

2. **Algorithm Execution**:
   - The code is set up to call either Dijkstra’s or Bellman-Ford algorithm, but these calls are commented out.

#### Why It’s Used:
- The main function is the entry point of the program. It reads the graph and prepares it for processing by one of the algorithms.

---

### **Summary**
This code is a powerful tool for solving shortest path problems in graphs. It uses **Dijkstra’s algorithm** for graphs with non-negative weights and **Bellman-Ford** for graphs that may contain negative weights. The main function reads the graph and prepares it for processing, while the algorithms compute the shortest paths.

In the next question, we can explore **potential improvements** to make the code more robust, efficient, and readable. Let me know if you’d like to proceed!