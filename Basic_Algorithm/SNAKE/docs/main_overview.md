# Code Overview: main.cpp

### Purpose of the Code

This C++ program simulates a **Snake Game**, a classic arcade game where a snake moves around a grid, eating food and growing longer. The game ends when the snake collides with itself or the boundaries of the grid. The purpose of this code is to simulate the game and determine how long the snake can survive before it collides with itself or the grid boundaries.

### Main Functionality

1. **Grid Representation**:
   - The game is played on an `n x n` grid.
   - The grid is represented by a 2D array `map[100][100]`, where `map[i][j]` is `true` if there is food at position `(i, j)` and `false` otherwise.

2. **Snake Movement**:
   - The snake starts at position `(0, 0)` and moves in one of four directions: up, right, down, or left.
   - The snake's direction is controlled by the `dir` variable, which corresponds to the indices of the `di` and `dj` arrays. These arrays represent the row and column changes for each direction.

3. **Food Consumption**:
   - When the snake moves to a cell containing food (`map[ci][cj] == true`), it consumes the food, and the snake grows longer.
   - The food is then removed from the grid (`map[ci][cj] = false`).

4. **Collision Detection**:
   - The game ends if the snake moves outside the grid boundaries (`ci < 0 || cj < 0 || ci >= n || cj >= n`) or if it collides with itself (`visit[ci][cj] == true`).

5. **Commands**:
   - The snake's direction can be changed at specific times using commands stored in a queue `q`. Each command specifies a time and a direction change (clockwise or counterclockwise).

6. **Time Tracking**:
   - The game keeps track of time using the variable `t`. The game ends when the snake collides, and the total time `t` is returned as the result.

### Algorithms Used

1. **Breadth-First Search (BFS) Inspired Movement**:
   - The snake's movement is simulated using a queue `snake` that stores the positions of the snake's body segments. The head of the snake is always at the front of the queue, and the tail is at the back.
   - When the snake moves, the new head position is added to the queue, and if no food is consumed, the tail position is removed from the queue.

2. **Direction Handling**:
   - The direction of the snake is updated based on the commands in the queue `q`. The direction is changed either clockwise or counterclockwise depending on the command.

3. **Collision Detection**:
   - The program uses a 2D array `visit[100][100]` to keep track of the positions occupied by the snake. If the snake moves to a position that is already marked as visited, a collision is detected, and the game ends.

### Overall Structure

1. **Initialization**:
   - The grid size `n` and the number of food items `k` are read from input.
   - The positions of the food items are stored in the `map` array.
   - The number of commands `l` and the commands themselves are read from input and stored in the queue `q`.

2. **Game Simulation**:
   - The `exec()` function simulates the game. It initializes the snake at position `(0, 0)` and starts moving the snake according to the current direction.
   - The function checks for commands at each time step and updates the snake's direction if necessary.
   - The function continues to move the snake until a collision is detected, at which point it returns the total time `t`.

3. **Output**:
   - The total time `t` is printed as the result, indicating how long the snake survived before the game ended.

### How the Different Parts of the Code Work Together

- **Global Variables**:
  - `map[100][100]`: Tracks the positions of food on the grid.
  - `visit[100][100]`: Tracks the positions occupied by the snake.
  - `di[4]` and `dj[4]`: Define the direction changes for up, right, down, and left.
  - `q`: A queue of commands that specify when and how the snake's direction should change.
  - `snake`: A queue that stores the positions of the snake's body segments.

- **`exec()` Function**:
  - This function is the core of the game simulation. It handles the snake's movement, direction changes, food consumption, and collision detection.
  - The function uses the `snake` queue to manage the snake's body and the `visit` array to detect collisions.

- **`main()` Function**:
  - This function initializes the game by reading input values, setting up the grid, and storing commands.
  - It then calls the `exec()` function to simulate the game and prints the result.

### Summary

This code simulates a Snake Game on an `n x n` grid. The snake moves around the grid, consuming food and growing longer. The game ends when the snake collides with itself or the grid boundaries. The program uses a queue to manage the snake's body and a 2D array to track visited positions for collision detection. The total time the snake survives is calculated and printed as the result.