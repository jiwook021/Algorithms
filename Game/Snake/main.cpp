// Snake Game with AI Player
// This program implements a classic snake game with an AI that plays automatically
// using the A* pathfinding algorithm to navigate toward food while avoiding collisions.

#include <iostream>
#include <vector>
#include <deque>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>  // Added missing header
#include <functional>
#include <atomic>
#include <string>

// Game constants
constexpr int GRID_WIDTH = 20;
constexpr int GRID_HEIGHT = 15;
constexpr int FRAME_DELAY_MS = 100; // Milliseconds between frames

// Enumerations for game elements and directions
enum class CellType { Empty, Snake, Food, Wall };
enum class Direction { Up, Right, Down, Left };

// 2D Position structure
struct Position {
    int x;
    int y;

    // Overload equality operator for comparison in algorithms
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }

    // Overload addition operator for vector math
    Position operator+(const Position& other) const {
        return { x + other.x, y + other.y };
    }

    // Hash function for Position to be used in unordered containers
    struct Hash {
        std::size_t operator()(const Position& pos) const {
            return std::hash<int>()(pos.x) ^ (std::hash<int>()(pos.y) << 1);
        }
    };
};

// Node for A* pathfinding algorithm
struct PathNode {
    Position pos;
    int g_cost; // Cost from start to current node
    int h_cost; // Heuristic cost (estimated cost from current to goal)
    Position parent; // Parent position for reconstructing the path

    // Calculate f_cost (total estimated cost)
    int f_cost() const { return g_cost + h_cost; }

    // Compare nodes based on f_cost for priority queue
    bool operator>(const PathNode& other) const {
        return f_cost() > other.f_cost();
    }
};

// Game state and logic
class SnakeGame {
private:
    // Game board representation
    std::vector<std::vector<CellType>> grid;

    // Snake representation: deque allows efficient operations at both ends
    std::deque<Position> snake;

    // Current movement direction
    Direction currentDirection;

    // Food position
    Position food;

    // Game state
    bool gameOver;
    int score;

    // Random number generator for food placement
    std::mt19937 rng;

    // Mutex for thread safety during updates
    mutable std::mutex gameMutex;

    // Atomic flag for game running state
    std::atomic<bool> isRunning;

public:
    // Direction vectors mapped to enum - made public for AI access
    const std::vector<Position> directionVectors = {
        {0, -1},  // Up
        {1, 0},   // Right
        {0, 1},   // Down
        {-1, 0}   // Left
    };

    // Constructor initializes game state
    SnakeGame() :
        grid(GRID_HEIGHT, std::vector<CellType>(GRID_WIDTH, CellType::Empty)),
        currentDirection(Direction::Right),
        gameOver(false),
        score(0),
        isRunning(true) {

        // Initialize random number generator with current time
        auto seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
        rng.seed(seed);

        // Initialize snake with a single segment in the middle of the grid
        snake.push_back({ GRID_WIDTH / 2, GRID_HEIGHT / 2 });
        grid[snake.front().y][snake.front().x] = CellType::Snake;

        // Place initial food
        placeFood();
    }

    // Destructor to ensure clean shutdown
    ~SnakeGame() {
        isRunning = false;
    }

    // Place food at a random empty position
    void placeFood() {
        // Find all empty cells
        std::vector<Position> emptyCells;
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            for (int x = 0; x < GRID_WIDTH; ++x) {
                if (grid[y][x] == CellType::Empty) {
                    emptyCells.push_back({ x, y });
                }
            }
        }

        // If there are no empty cells, game is complete
        if (emptyCells.empty()) {
            gameOver = true;
            return;
        }

        // Pick a random empty cell for the food
        std::uniform_int_distribution<int> dist(0, static_cast<int>(emptyCells.size() - 1));
        food = emptyCells[dist(rng)];
        grid[food.y][food.x] = CellType::Food;
    }

    // Update game state for one step
    bool update(Direction newDirection) {
        // Lock during update for thread safety
        std::lock_guard<std::mutex> lock(gameMutex);

        if (gameOver) {
            return false;
        }

        // Prevent 180-degree turns (snake can't turn directly back on itself)
        if ((currentDirection == Direction::Up && newDirection == Direction::Down) ||
            (currentDirection == Direction::Down && newDirection == Direction::Up) ||
            (currentDirection == Direction::Left && newDirection == Direction::Right) ||
            (currentDirection == Direction::Right && newDirection == Direction::Left)) {
            newDirection = currentDirection;
        }

        currentDirection = newDirection;

        // Calculate new head position based on direction
        Position newHead = snake.front() + directionVectors[static_cast<int>(currentDirection)];

        // Check for collisions with walls
        if (newHead.x < 0 || newHead.x >= GRID_WIDTH || newHead.y < 0 || newHead.y >= GRID_HEIGHT) {
            gameOver = true;
            return false;
        }

        // Check for collision with snake body (except tail which will move)
        if (grid[newHead.y][newHead.x] == CellType::Snake &&
            !(newHead.x == snake.back().x && newHead.y == snake.back().y)) {
            gameOver = true;
            return false;
        }

        // Check if food is eaten
        bool foodEaten = (grid[newHead.y][newHead.x] == CellType::Food);

        // Move snake: add new head
        snake.push_front(newHead);
        grid[newHead.y][newHead.x] = CellType::Snake;

        // If food wasn't eaten, remove tail
        if (!foodEaten) {
            Position tail = snake.back();
            grid[tail.y][tail.x] = CellType::Empty;
            snake.pop_back();
        }
        else {
            // Food was eaten, increase score and place new food
            score++;
            placeFood();
        }

        return true;
    }

    // Render the current game state to the console
    void render() {
        // Lock during rendering to prevent concurrent modification
        std::lock_guard<std::mutex> lock(gameMutex);

        // Clear console (platform dependent)
        std::cout << "\033[2J\033[1;1H";  // ANSI escape sequence to clear screen

        // Draw top border
        std::cout << "+" << std::string(GRID_WIDTH * 2, '-') << "+\n";

        // Draw grid
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            std::cout << "|";
            for (int x = 0; x < GRID_WIDTH; ++x) {
                // Compare with explicit enumeration values to avoid ambiguity
                if (grid[y][x] == CellType::Empty) {
                    std::cout << "  ";
                }
                else if (grid[y][x] == CellType::Snake) {
                    if (x == snake.front().x && y == snake.front().y) {
                        std::cout << "HH";  // Snake head (using ASCII instead of Unicode)
                    }
                    else {
                        std::cout << "OO";  // Snake body (using ASCII instead of Unicode)
                    }
                }
                else if (grid[y][x] == CellType::Food) {
                    std::cout << "XX";
                }
                else if (grid[y][x] == CellType::Wall) {
                    std::cout << "##";
                }
            }
            std::cout << "|\n";
        }

        // Draw bottom border
        std::cout << "+" << std::string(GRID_WIDTH * 2, '-') << "+\n";

        // Display score
        std::cout << "Score: " << score << std::endl;

        // Display game over message if applicable
        if (gameOver) {
            std::cout << "Game Over!" << std::endl;
        }
    }

    // Getters for AI use

    const std::deque<Position>& getSnake() const {
        return snake;
    }

    Position getFood() const {
        return food;
    }

    const std::vector<std::vector<CellType>>& getGrid() const {
        return grid;
    }

    bool isGameOver() const {
        return gameOver;
    }

    int getScore() const {
        return score;
    }

    bool getRunningState() const {
        return isRunning;
    }

    void setRunningState(bool state) {
        isRunning = state;
    }
};

// AI Controller for the snake
class SnakeAI {
private:
    const SnakeGame& game;

    // Calculate Manhattan distance between two positions (heuristic for A*)
    int calculateManhattanDistance(const Position& start, const Position& end) const {
        return std::abs(start.x - end.x) + std::abs(start.y - end.y);
    }

    // Check if a position is valid for movement (not a wall or snake body)
    bool isValidPosition(const Position& pos) const {
        // Check bounds
        if (pos.x < 0 || pos.x >= GRID_WIDTH || pos.y < 0 || pos.y >= GRID_HEIGHT) {
            return false;
        }

        const auto& grid = game.getGrid();

        // To avoid operator ambiguity, use explicit comparison to enum values
        bool isSnake = (grid[pos.y][pos.x] == CellType::Snake);
        bool isTail = (pos.x == game.getSnake().back().x && pos.y == game.getSnake().back().y);
        bool isFood = (pos.x == game.getFood().x && pos.y == game.getFood().y);

        // Valid if not snake OR is the tail (which will move) AND not food
        return !isSnake || (isTail && !isFood);
    }

    // Path finding result structure - replacement for std::optional
    struct PathResult {
        bool found;
        std::vector<Position> path;

        // Constructor for successful path finding
        PathResult(std::vector<Position> p) : found(true), path(std::move(p)) {}

        // Constructor for unsuccessful path finding
        PathResult() : found(false) {}

        // Check if path was found
        operator bool() const { return found; }
    };

    // A* pathfinding algorithm to find path to food
    PathResult findPathToFood() const {
        const Position& start = game.getSnake().front();  // Snake head
        const Position& goal = game.getFood();            // Food position

        // If start and goal are the same, return empty path
        if (start.x == goal.x && start.y == goal.y) {
            return PathResult(std::vector<Position>{});
        }

        // Open set (nodes to be evaluated)
        std::priority_queue<PathNode, std::vector<PathNode>, std::greater<PathNode>> openSet;

        // Closed set (nodes already evaluated)
        std::unordered_set<Position, Position::Hash> closedSet;

        // Using explicit maps to store costs and parents
        std::unordered_map<Position, int, Position::Hash> gCosts;
        std::unordered_map<Position, Position, Position::Hash> parents;

        // Initialize with start node
        openSet.push({
            start,
            0,
            calculateManhattanDistance(start, goal),
            {-1, -1}  // Invalid parent for start
            });

        // Set initial g_cost
        gCosts[start] = 0;

        // Main A* loop
        while (!openSet.empty()) {
            // Get node with lowest f_cost
            PathNode current = openSet.top();
            openSet.pop();

            // If we reached the goal, reconstruct and return the path
            if (current.pos.x == goal.x && current.pos.y == goal.y) {
                std::vector<Position> path;
                Position pos = current.pos;

                // Don't include start position in path
                while (!(pos.x == start.x && pos.y == start.y)) {
                    path.push_back(pos);
                    pos = parents[pos];
                }

                // Reverse to get path from start to goal
                std::reverse(path.begin(), path.end());
                return PathResult(path);
            }

            // Add current to closed set
            closedSet.insert(current.pos);

            // Check all four neighboring positions
            const std::vector<Position> directions = {
                {0, -1}, {1, 0}, {0, 1}, {-1, 0}
            };

            for (const auto& dir : directions) {
                Position neighborPos = current.pos + dir;

                // Skip if closed or not valid
                auto it = closedSet.find(neighborPos);
                if (it != closedSet.end() || !isValidPosition(neighborPos)) {
                    continue;
                }

                // Calculate g_cost (distance from start)
                int tentativeGCost = gCosts[current.pos] + 1;

                // If neighbor not in open set or has better g_cost
                auto costIt = gCosts.find(neighborPos);
                if (costIt == gCosts.end() || tentativeGCost < costIt->second) {
                    // Update cost and parent
                    gCosts[neighborPos] = tentativeGCost;
                    parents[neighborPos] = current.pos;

                    // Calculate h_cost (heuristic to goal)
                    int hCost = calculateManhattanDistance(neighborPos, goal);

                    // Add to open set
                    openSet.push({
                        neighborPos,
                        tentativeGCost,
                        hCost,
                        current.pos
                        });
                }
            }
        }

        // No path found
        return PathResult();
    }

    // Fallback strategy when no direct path to food is available
    Direction findSafestDirection() const {
        const Position& head = game.getSnake().front();
        std::vector<Direction> safeDirections;

        // Check each direction
        for (int i = 0; i < 4; ++i) {
            Direction dir = static_cast<Direction>(i);
            Position newPos = head + game.directionVectors[i];

            if (isValidPosition(newPos)) {
                safeDirections.push_back(dir);
            }
        }

        // If no safe directions, return current direction (will likely cause game over)
        if (safeDirections.empty()) {
            return Direction::Right;  // Arbitrary choice when no options
        }

        // Prefer directions that move away from snake body and closer to food
        std::sort(safeDirections.begin(), safeDirections.end(), [&](Direction a, Direction b) {
            Position posA = head + game.directionVectors[static_cast<int>(a)];
            Position posB = head + game.directionVectors[static_cast<int>(b)];

            // Calculate distance to food for each position
            int distA = calculateManhattanDistance(posA, game.getFood());
            int distB = calculateManhattanDistance(posB, game.getFood());

            return distA < distB;  // Prefer shorter distance to food
            });

        return safeDirections.front();
    }

public:
    // Constructor
    explicit SnakeAI(const SnakeGame& gameRef) : game(gameRef) {}

    // Calculate next move
    Direction getNextMove() const {
        // Try to find path to food
        auto path = findPathToFood();

        // If path exists and has at least one step
        if (path && !path.path.empty()) {
            Position nextPos = path.path[0];
            Position head = game.getSnake().front();

            // Determine direction to next position
            if (nextPos.x > head.x) return Direction::Right;
            if (nextPos.x < head.x) return Direction::Left;
            if (nextPos.y > head.y) return Direction::Down;
            if (nextPos.y < head.y) return Direction::Up;
        }

        // If no path found, use fallback strategy
        return findSafestDirection();
    }
};

// Main game loop with AI
void runGame() {
    // Initialize game
    SnakeGame game;
    
    SnakeAI ai(game);

    // Game loop
    while (game.getRunningState() && !game.isGameOver()) {
        // Get AI's next move
        Direction nextMove = ai.getNextMove();

        // Update game with AI's move
        if (!game.update(nextMove)) {
            break;  // Game over or error
        }

        // Render current state
        game.render();

        // Delay for visualization
        std::this_thread::sleep_for(std::chrono::milliseconds(FRAME_DELAY_MS));
    }

    // Game over
    game.render();
    std::cout << "Final Score: " << game.getScore() << std::endl;
}

// Entry point
int main() {
    try {
        std::cout << "Snake Game with AI Player" << std::endl;
        std::cout << "Press Ctrl+C to exit" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));

        runGame();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 2;
    }

    return 0;
}