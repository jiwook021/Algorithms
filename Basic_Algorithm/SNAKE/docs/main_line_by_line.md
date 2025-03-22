# Step-by-Step Explanation: main.cpp

Let’s break down the code **step by step** in a way that’s easy to understand, even for someone who is just starting to learn programming. I’ll explain each part of the code, the logic behind it, and why certain techniques are used. I’ll also include examples and diagrams where helpful.

---

### **1. Header Files and Namespace**
```cpp
#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
using namespace std;
```

#### What it does:
- **`#include` statements**: These bring in external libraries that provide useful functions and data structures.
  - `<algorithm>`: Provides functions like sorting and searching.
  - `<iostream>`: Used for input and output (e.g., `cin` and `cout`).
  - `<vector>`: Provides a dynamic array (not used in this code but included).
  - `<queue>`: Provides the `queue` data structure, which is used to store commands and the snake’s body.
- **`using namespace std`**: This allows us to use standard library functions (like `cin` and `cout`) without typing `std::` every time.

#### Why it’s used:
- These libraries are included to make the code easier to write and more efficient. For example, the `queue` data structure is essential for managing the snake’s body and commands.

---

### **2. Global Variables**
```cpp
typedef long long ll;
const int INF = 987654321;
int n, m, ci = 0, cj = 0, k, tailI = 0, tailJ = 0;
bool map[100][100];
int visit[100][100];
int di[4] = {-1, 0, 1, 0};
int dj[4] = {0, 1, 0, -1};
int dir = 1;
```

#### What it does:
- **`typedef long long ll`**: Defines a shortcut for the `long long` data type, which can store very large integers.
- **`const int INF = 987654321`**: Defines a constant value `INF` (infinity), which is often used in algorithms to represent an unreachable or invalid value.
- **`int n, m, ci, cj, k, tailI, tailJ`**: These are variables used to store:
  - `n`: The size of the grid (n x n).
  - `m`: Not used in this code (likely a leftover).
  - `ci`, `cj`: The current position of the snake’s head.
  - `k`: The number of food items on the grid.
  - `tailI`, `tailJ`: The position of the snake’s tail (not used in this code).
- **`bool map[100][100]`**: A 2D array representing the grid. `map[i][j]` is `true` if there’s food at position `(i, j)`.
- **`int visit[100][100]`**: A 2D array to track which positions the snake has visited (to detect collisions).
- **`int di[4]`, `int dj[4]`**: Arrays that define the direction changes for the snake:
  - `di`: Changes in the row index for up, right, down, left.
  - `dj`: Changes in the column index for up, right, down, left.
- **`int dir = 1`**: The current direction of the snake. Initially, it’s set to `1`, which corresponds to moving right.

#### Why it’s used:
- Global variables are used here to make the code simpler and avoid passing many parameters between functions. However, this is generally not recommended in larger programs because it can make the code harder to debug and maintain.

---

### **3. Command Structure**
```cpp
struct command {
  int time;
  bool clock;
  command(int t, bool c): time(t), clock(c) {}
};
queue<command> q;
```

#### What it does:
- **`struct command`**: Defines a custom data type called `command`. Each `command` has two properties:
  - `time`: The time at which the direction change should happen.
  - `clock`: A boolean value (`true` for clockwise, `false` for counterclockwise).
- **`queue<command> q`**: A queue that stores all the commands. Queues are **First-In-First-Out (FIFO)**, meaning the first command added is the first one processed.

#### Why it’s used:
- The queue is used to store and process commands in the order they are received. This ensures that the snake changes direction at the correct times.

---

### **4. Snake Queue**
```cpp
queue<pair<int, int>> snake;
```

#### What it does:
- **`queue<pair<int, int>> snake`**: A queue that stores the positions of the snake’s body segments. Each element is a `pair<int, int>`, representing the row and column indices of a segment.

#### Why it’s used:
- The queue is used to manage the snake’s body. When the snake moves, the new head position is added to the front of the queue, and the tail position is removed from the back (if no food is eaten).

---

### **5. `exec()` Function**
```cpp
int exec() {
  int t = 1;
  snake.push(make_pair(0, 0));
  visit[0][0] = true;
  while(true) {
    if(!q.empty() && q.front().time == t - 1) {
      if(q.front().clock && ++dir == 4) dir = 0;
      if(!q.front().clock && --dir == -1) dir = 3;
      q.pop();
    }
    ci += di[dir], cj += dj[dir];
    if(ci < 0 || cj < 0 || ci >= n || cj >= n || visit[ci][cj]) break;
    visit[ci][cj] = true;
    snake.push(make_pair(ci, cj));
    if(!map[ci][cj]) {
      pair<int, int> p = snake.front();
      snake.pop();
      visit[p.first][p.second] = false;
    } else map[ci][cj] = false;
    t++;
  }
  return t;
}
```

#### What it does:
1. **Initialization**:
   - `t = 1`: Starts tracking time.
   - `snake.push(make_pair(0, 0))`: Places the snake’s head at position `(0, 0)`.
   - `visit[0][0] = true`: Marks the starting position as visited.

2. **Game Loop**:
   - The `while(true)` loop runs until the snake collides.
   - **Direction Change**:
     - If a command is due (`q.front().time == t - 1`), the snake’s direction is updated.
     - `dir` is incremented (clockwise) or decremented (counterclockwise), wrapping around if necessary.
   - **Move Snake**:
     - The head moves to a new position `(ci, cj)` based on the current direction.
   - **Collision Check**:
     - If the new position is outside the grid or already visited, the game ends.
   - **Update Snake**:
     - The new head position is added to the `snake` queue.
     - If no food is eaten, the tail is removed from the queue and its position is marked as unvisited.
   - **Food Check**:
     - If food is eaten, it is removed from the grid.

3. **Return Time**:
   - The function returns the total time `t` when the game ends.

#### Why it’s used:
- This function simulates the snake’s movement and handles all game logic, including direction changes, collision detection, and food consumption.

---

### **6. `main()` Function**
```cpp
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0); cout.tie(0);
  cin >> n >> k;
  for(int i = 0; i < k; i++) {
    int a, b; cin >> a >> b;
    map[a-1][b-1] = true;
  }
  int l; cin >> l;
  for(int i = 0; i < l; i++) {
    int x; char c; cin >> x >> c;
    q.push(command(x, c == 'D'));
  }
  cout << exec();
  return 0;
}
```

#### What it does:
1. **Input Setup**:
   - `ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);`: Speeds up input/output operations.
   - `cin >> n >> k`: Reads the grid size and number of food items.
   - The `for` loop reads the positions of the food items and marks them on the grid.

2. **Command Input**:
   - `cin >> l`: Reads the number of commands.
   - The `for` loop reads each command and adds it to the queue `q`.

3. **Game Execution**:
   - `cout << exec();`: Calls the `exec()` function to simulate the game and prints the result.

#### Why it’s used:
- The `main()` function handles input, sets up the game, and starts the simulation.

---

### **Text-Based Diagram of the Snake’s Movement**
```
Initial State:
Grid: 3x3
Food: (1, 1)
Snake: [(0, 0)]

Time 1:
Snake moves right to (0, 1)
Snake: [(0, 0), (0, 1)]

Time 2:
Snake moves right to (0, 2)
Snake: [(0, 0), (0, 1), (0, 2)]

Time 3:
Snake moves down to (1, 2)
Snake: [(0, 1), (0, 2), (1, 2)]
```

This diagram shows how the snake grows and moves over time.

---

### **Summary**
This code simulates a Snake Game using queues to manage the snake’s body and commands. The `exec()` function handles the game logic, while the `main()` function sets up the game and starts the simulation. The use of queues and 2D arrays makes the code efficient and easy to understand.