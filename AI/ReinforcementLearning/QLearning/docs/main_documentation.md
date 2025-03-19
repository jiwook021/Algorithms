# Documentation for `main.cpp`

## Overview

The `main.cpp` file implements a simple reinforcement learning environment using a grid world model. The primary purpose of this code is to simulate an agent navigating through a grid to reach a goal while avoiding obstacles. It demonstrates key concepts of reinforcement learning, such as state representation, action selection, and environment interaction. The code is structured to be extensible and reusable, providing a foundation for more complex reinforcement learning experiments.

## Key Components

### 1. `RLException` Class

#### Purpose
- A custom exception class derived from `std::runtime_error` to handle errors specific to the reinforcement learning environment.

#### Implementation
```cpp
class RLException : public std::runtime_error {
public:
    explicit RLException(const std::string& message) : std::runtime_error(message) {}
};
```

### 2. `State` Struct

#### Purpose
- Represents a position in the grid world with `x` and `y` coordinates.

#### Key Methods
- `bool operator==(const State& other) const`: Compares two states for equality.
- `std::string toString() const`: Returns a string representation of the state.

#### Implementation
```cpp
struct State {
    int x;
    int y;
    
    bool operator==(const State& other) const {
        return x == other.x && y == other.y;
    }
    
    std::string toString() const {
        std::stringstream ss;
        ss << "(" << x << "," << y << ")";
        return ss.str();
    }
};
```

### 3. `Action` Enum

#### Purpose
- Enumerates possible actions an agent can take: `UP`, `RIGHT`, `DOWN`, `LEFT`.

#### Helper Function
- `std::string actionToString(Action action)`: Converts an action to its string representation.

#### Implementation
```cpp
enum class Action {
    UP,
    RIGHT,
    DOWN,
    LEFT
};

std::string actionToString(Action action) {
    switch (action) {
        case Action::UP: return "UP";
        case Action::RIGHT: return "RIGHT";
        case Action::DOWN: return "DOWN";
        case Action::LEFT: return "LEFT";
        default: throw RLException("Invalid action");
    }
}
```

### 4. `Environment` Interface

#### Purpose
- Defines the interface for any reinforcement learning environment.

#### Key Methods
- `virtual State reset() = 0`: Resets the environment to the initial state.
- `virtual std::tuple<State, double, bool> step(const Action& action) = 0`: Executes an action and returns the next state, reward, and whether the episode is done.
- `virtual State getCurrentState() const = 0`: Returns the current state.
- `virtual bool isValidAction(const Action& action) const = 0`: Checks if an action is valid in the current state.
- `virtual std::vector<Action> getValidActions() const = 0`: Returns valid actions for the current state.
- `virtual void render() const = 0`: Visualizes the environment.

### 5. `GridWorld` Class

#### Purpose
- Implements the `Environment` interface for a grid world where an agent navigates to reach a goal while avoiding obstacles.

#### Key Attributes
- `int width, height`: Dimensions of the grid.
- `State currentState, initialState, goalState`: States representing the current, initial, and goal positions.
- `std::vector<State> obstacles`: Positions of obstacles in the grid.
- `std::mutex mutex`: Ensures thread-safe operations.

#### Key Methods
- Constructor: Initializes the grid world with dimensions, initial state, goal state, and obstacles. Validates input parameters.
- Deleted copy constructor and assignment operator to prevent accidental copying.

#### Implementation
```cpp
class GridWorld : public Environment {
private:
    int width;
    int height;
    State currentState;
    State initialState;
    State goalState;
    std::vector<State> obstacles;
    mutable std::mutex mutex;

public:
    GridWorld(int width, int height, State initialState, State goalState, const std::vector<State>& obstacles);
    // Other methods...
};
```

## Algorithm Analysis

- **Complexity**: The complexity of operations such as checking valid actions or moving to a new state is generally constant, O(1), due to the fixed size of the grid and the use of efficient data structures like `unordered_map`.
- **Approach**: The code uses a straightforward grid-based approach to model the environment, which is common in reinforcement learning for its simplicity and clarity.

## Dependencies and Interactions

- **Standard Libraries**: Utilizes various C++ standard libraries such as `<iostream>`, `<vector>`, `<unordered_map>`, and `<mutex>` for basic operations, data storage, and thread safety.
- **Custom Hash Function**: Implements a custom hash function for the `State` struct to enable its use as a key in `unordered_map`.

## Usage Example

```cpp
int main() {
    std::cout << "Reinforcement Learning Demo: Q-Learning in Grid World\n";
    runReinforcementLearningDemo();
    return 0;
}
```

## Potential Issues, Edge Cases, and Limitations

- **Invalid Parameters**: The constructor of `GridWorld` throws exceptions if the grid dimensions are non-positive or if states are out of bounds or overlap with obstacles.
- **Thread Safety**: While a mutex is used, the current implementation may not fully support concurrent operations if extended beyond the current scope.
- **Action Validation**: The code assumes actions are always valid when passed to the `step` function; additional checks may be needed for robustness in more complex scenarios.

This documentation provides a comprehensive understanding of the `main.cpp` file, aiding developers in using and extending the code for reinforcement learning experiments.