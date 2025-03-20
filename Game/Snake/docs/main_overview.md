# Code Overview: main.cpp

This C++ code implements a classic Snake game with an AI player that uses the A* pathfinding algorithm to navigate the snake toward food while avoiding collisions. Let's break down the purpose, functionality, and structure of the code in detail.

### Purpose and Main Functionality

The primary purpose of this code is to create a Snake game where the snake is controlled by an AI rather than a human player. The AI uses the A* pathfinding algorithm to determine the optimal path to the food while avoiding walls and the snake's own body. The game runs in a grid-based environment, and the snake grows in length each time it eats food, increasing the difficulty over time.

### Problem Being Solved

The problem being solved is twofold:
1. **Game Logic**: Implementing the core mechanics of the Snake game, including snake movement, food placement, collision detection, and game state management.
2. **AI Navigation**: Implementing an AI that can autonomously control the snake, making decisions to navigate toward food while avoiding obstacles.

### Approach Taken

The code uses several key components and algorithms to achieve its goals:

1. **Game Board Representation**:
   - The game board is represented as a 2D grid (`std::vector<std::vector<CellType>> grid`), where each cell can be empty, part of the snake, food, or a wall.
   - The snake is represented as a deque (`std::deque<Position> snake`), which allows efficient addition and removal of elements at both ends.

2. **A* Pathfinding Algorithm**:
   - The AI uses the A* algorithm to find the shortest path from the snake's head to the food.
   - The algorithm uses a priority queue (`std::priority_queue<PathNode>`) to explore nodes based on their total estimated cost (`f_cost`), which is the sum of the cost from the start (`g_cost`) and the heuristic cost to the goal (`h_cost`).

3. **Thread Safety and Synchronization**:
   - The game uses a mutex (`std::mutex gameMutex`) to ensure thread-safe updates to the game state.
   - An atomic flag (`std::atomic<bool> isRunning`) is used to manage the game's running state.

4. **Random Number Generation**:
   - A Mersenne Twister random number generator (`std::mt19937 rng`) is used to place food at random positions on the grid.

### Overall Structure

The code is structured into several key parts:

1. **Enumerations and Structures**:
   - `CellType` and `Direction` enumerations define the types of cells and possible directions.
   - `Position` structure represents a 2D position on the grid, with overloaded operators for comparison and vector math.
   - `PathNode` structure is used by the A* algorithm to represent nodes in the search space.

2. **SnakeGame Class**:
   - This class encapsulates the game state and logic, including the grid, snake, food, and AI navigation.
   - It provides methods for initializing the game, updating the game state, and handling AI decisions.

3. **Main Function**:
   - The `main` function serves as the entry point of the program.
   - It initializes the game, handles exceptions, and starts the game loop.

### How Different Parts Work Together

1. **Initialization**:
   - The `SnakeGame` constructor initializes the game state, including the grid, snake, and food positions.
   - The random number generator is seeded to ensure different food placements in each game.

2. **Game Loop**:
   - The game runs in a loop, updating the game state at regular intervals (`FRAME_DELAY_MS`).
   - The AI uses the A* algorithm to determine the next move for the snake.

3. **AI Navigation**:
   - The AI calculates the shortest path to the food using the A* algorithm.
   - It updates the snake's direction based on the calculated path.

4. **Collision Detection and Game Over**:
   - The game checks for collisions with walls or the snake's own body.
   - If a collision is detected, the game ends, and the final score is displayed.

5. **Thread Safety**:
   - The mutex ensures that updates to the game state are thread-safe, preventing race conditions.
   - The atomic flag manages the game's running state, allowing safe termination of the game loop.

### Summary

This code implements a Snake game with an AI player that uses the A* pathfinding algorithm to navigate the snake toward food while avoiding collisions. The game is structured around a grid-based representation, with the snake's movements and AI decisions managed by the `SnakeGame` class. The use of thread-safe mechanisms ensures smooth and reliable game updates, while the A* algorithm provides efficient and intelligent pathfinding for the AI.