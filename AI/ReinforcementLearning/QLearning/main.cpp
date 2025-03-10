#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <optional>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <concepts>
#include <sstream>
#include <iomanip>
#include <thread>

/**
 * @brief Custom exception class for reinforcement learning errors
 */
class RLException : public std::runtime_error {
public:
    explicit RLException(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Represents a state in the environment
 */
struct State {
    int x;
    int y;
    
    // Define equality operator for using State in containers
    bool operator==(const State& other) const {
        return x == other.x && y == other.y;
    }
    
    // For string representation
    std::string toString() const {
        std::stringstream ss;
        ss << "(" << x << "," << y << ")";
        return ss.str();
    }
};

// Custom hash function for State to use it as key in unordered_map
namespace std {
    template <>
    struct hash<State> {
        size_t operator()(const State& state) const {
            // Simple hash function for State
            return hash<int>()(state.x) ^ (hash<int>()(state.y) << 1);
        }
    };
}

/**
 * @brief Represents an action that an agent can take
 */
enum class Action {
    UP,
    RIGHT,
    DOWN,
    LEFT
};

// Helper function to convert Action to string
std::string actionToString(Action action) {
    switch (action) {
        case Action::UP: return "UP";
        case Action::RIGHT: return "RIGHT";
        case Action::DOWN: return "DOWN";
        case Action::LEFT: return "LEFT";
        default: throw RLException("Invalid action");
    }
}

// Vector of all possible actions for easy iteration
const std::vector<Action> ALL_ACTIONS = {
    Action::UP, Action::RIGHT, Action::DOWN, Action::LEFT
};

/**
 * @brief Interface for environment in reinforcement learning
 */
class Environment {
public:
    virtual ~Environment() = default;
    
    // Reset environment to initial state
    virtual State reset() = 0;
    
    // Take an action and return next state, reward, and if episode is done
    virtual std::tuple<State, double, bool> step(const Action& action) = 0;
    
    // Get current state
    virtual State getCurrentState() const = 0;
    
    // Check if action is valid for current state
    virtual bool isValidAction(const Action& action) const = 0;
    
    // Get available actions for current state
    virtual std::vector<Action> getValidActions() const = 0;
    
    // Visualize the environment
    virtual void render() const = 0;
};

/**
 * @brief Grid world environment implementation
 * 
 * Represents a grid world where agent navigates to reach a goal
 * while avoiding obstacles.
 */
class GridWorld : public Environment {
private:
    int width;
    int height;
    State currentState;
    State initialState;  // Store initial state separately
    State goalState;
    std::vector<State> obstacles;
    mutable std::mutex mutex; // Simplified to regular mutex

public:
    /**
     * @brief Construct a new Grid World environment
     * 
     * @param width Width of the grid
     * @param height Height of the grid
     * @param initialState Starting position
     * @param goalState Target position
     * @param obstacles Vector of obstacle positions
     * 
     * @throws RLException if parameters are invalid
     */
    GridWorld(
        int width,
        int height,
        State initialState,
        State goalState,
        const std::vector<State>& obstacles
    ) : width(width), height(height), 
        currentState(initialState),
        initialState(initialState),
        goalState(goalState),
        obstacles(obstacles) {
        
        // Validate parameters
        if (width <= 0 || height <= 0) {
            throw RLException("Grid dimensions must be positive");
        }
        
        // Check if initial or goal state is within boundaries
        if (initialState.x < 0 || initialState.x >= width || 
            initialState.y < 0 || initialState.y >= height) {
            throw RLException("Initial state outside grid boundaries");
        }
        
        if (goalState.x < 0 || goalState.x >= width || 
            goalState.y < 0 || goalState.y >= height) {
            throw RLException("Goal state outside grid boundaries");
        }
        
        // Check if initial or goal state is an obstacle
        for (const auto& obstacle : obstacles) {
            if (initialState == obstacle || goalState == obstacle) {
                throw RLException("Initial or goal state cannot be an obstacle");
            }
            
            if (obstacle.x < 0 || obstacle.x >= width || 
                obstacle.y < 0 || obstacle.y >= height) {
                throw RLException("Obstacle outside grid boundaries");
            }
        }
    }

    // No copy constructor to prevent accidental copies
    GridWorld(const GridWorld&) = delete;
    GridWorld& operator=(const GridWorld&) = delete;

    // Move constructor and assignment allowed
    GridWorld(GridWorld&&) = default;
    GridWorld& operator=(GridWorld&&) = default;

    /**
     * @brief Reset the environment to initial state
     * 
     * @return State The initial state
     */
    State reset() override {
        std::lock_guard<std::mutex> lock(mutex);
        currentState = initialState; // Reset to initial position
        return currentState;
    }

    /**
     * @brief Take an action and return next state, reward, done flag
     * 
     * @param action The action to take
     * @return tuple containing next state, reward, and done flag
     */
    std::tuple<State, double, bool> step(const Action& action) override {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Make a copy of current state for calculations
        State nextState = currentState;
        
        // Calculate next state based on action
        switch (action) {
            case Action::UP:
                nextState.y = std::max(0, nextState.y - 1);
                break;
            case Action::RIGHT:
                nextState.x = std::min(width - 1, nextState.x + 1);
                break;
            case Action::DOWN:
                nextState.y = std::min(height - 1, nextState.y + 1);
                break;
            case Action::LEFT:
                nextState.x = std::max(0, nextState.x - 1);
                break;
        }
        
        // Check if next state is an obstacle
        if (std::find(obstacles.begin(), obstacles.end(), nextState) != obstacles.end()) {
            // Stay in current state if would hit obstacle
            return {currentState, -1.0, false}; // Penalty for hitting obstacle
        }
        
        // Update current state
        currentState = nextState;
        
        // Check if reached goal
        bool done = (currentState == goalState);
        double reward = done ? 10.0 : -0.1; // Reward for goal, small penalty for each step
        
        return {currentState, reward, done};
    }

    /**
     * @brief Get the current state
     * 
     * @return State The current state
     */
    State getCurrentState() const override {
        std::lock_guard<std::mutex> lock(mutex);
        return currentState;
    }

    /**
     * @brief Check if action is valid for current state
     * 
     * @param action The action to check
     * @return bool True if action is valid
     */
    bool isValidAction(const Action& action) const override {
        std::lock_guard<std::mutex> lock(mutex);
        State nextState = currentState;
        
        // Calculate next state based on action
        switch (action) {
            case Action::UP:
                nextState.y = std::max(0, nextState.y - 1);
                break;
            case Action::RIGHT:
                nextState.x = std::min(width - 1, nextState.x + 1);
                break;
            case Action::DOWN:
                nextState.y = std::min(height - 1, nextState.y + 1);
                break;
            case Action::LEFT:
                nextState.x = std::max(0, nextState.x - 1);
                break;
        }
        
        return nextState.x >= 0 && nextState.x < width && 
               nextState.y >= 0 && nextState.y < height;
    }

    /**
     * @brief Get all valid actions for current state
     * 
     * @return vector of valid actions
     */
    std::vector<Action> getValidActions() const override {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<Action> validActions;
        
        for (const auto& action : ALL_ACTIONS) {
            State nextState = currentState;
            
            // Calculate next state directly
            switch (action) {
                case Action::UP:
                    nextState.y = std::max(0, nextState.y - 1);
                    break;
                case Action::RIGHT:
                    nextState.x = std::min(width - 1, nextState.x + 1);
                    break;
                case Action::DOWN:
                    nextState.y = std::min(height - 1, nextState.y + 1);
                    break;
                case Action::LEFT:
                    nextState.x = std::max(0, nextState.x - 1);
                    break;
            }
            
            // Check if next state is valid
            if (nextState.x >= 0 && nextState.x < width && 
                nextState.y >= 0 && nextState.y < height) {
                validActions.push_back(action);
            }
        }
        
        return validActions;
    }

    /**
     * @brief Visualize the grid world
     */
    void render() const override {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Create a grid representation
        std::vector<std::vector<char>> grid(height, std::vector<char>(width, '.'));
        
        // Mark obstacles
        for (const auto& obstacle : obstacles) {
            if (obstacle.x >= 0 && obstacle.x < width && 
                obstacle.y >= 0 && obstacle.y < height) {
                grid[obstacle.y][obstacle.x] = '#';
            }
        }
        
        // Mark goal
        grid[goalState.y][goalState.x] = 'G';
        
        // Mark current position
        grid[currentState.y][currentState.x] = 'A';
        
        // Print the grid
        std::cout << "\n";
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                std::cout << grid[y][x] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << "\n";
    }
};

/**
 * @brief Interface for reinforcement learning agents
 */
template <typename S, typename A>
class RLAgent {
public:
    virtual ~RLAgent() = default;
    
    // Choose an action based on current state
    virtual A chooseAction(const S& state) = 0;
    
    // Update the agent's knowledge based on experience
    virtual void update(const S& state, const A& action, double reward, const S& nextState, bool done) = 0;
    
    // Save the learned policy/value function
    virtual void savePolicy(const std::string& filename) const = 0;
    
    // Load a previously learned policy/value function
    virtual void loadPolicy(const std::string& filename) = 0;
};

/**
 * @brief Q-Learning agent implementation
 * 
 * Q-Learning is a model-free reinforcement learning algorithm
 * that learns a policy telling an agent what action to take under what circumstances.
 */
class QLearningAgent : public RLAgent<State, Action> {
private:
    // Q-Table: maps state-action pairs to Q-values
    std::unordered_map<State, std::unordered_map<Action, double>> qTable;
    
    double learningRate;
    double discountFactor;
    double explorationRate;
    double explorationDecay;
    double minExplorationRate;
    
    // Random generator as a shared pointer to avoid const issues
    std::shared_ptr<std::mt19937> randomGenerator;
    mutable std::mutex mutex; // For thread-safety

public:
    /**
     * @brief Construct a new Q-Learning Agent
     * 
     * @param learningRate Alpha - how quickly agent incorporates new information (0-1)
     * @param discountFactor Gamma - importance of future rewards (0-1)
     * @param explorationRate Epsilon - probability of random action (0-1)
     * @param explorationDecay Rate at which exploration rate decreases
     * @param minExplorationRate Minimum exploration rate
     * @param seed Random seed for reproducibility
     * 
     * @throws RLException if parameters are invalid
     */
    QLearningAgent(
        double learningRate = 0.1,
        double discountFactor = 0.99,
        double explorationRate = 1.0,
        double explorationDecay = 0.995,
        double minExplorationRate = 0.01,
        unsigned int seed = std::random_device{}()
    ) : learningRate(learningRate),
        discountFactor(discountFactor),
        explorationRate(explorationRate),
        explorationDecay(explorationDecay),
        minExplorationRate(minExplorationRate),
        randomGenerator(std::make_shared<std::mt19937>(seed)) {
        
        // Validate parameters
        if (learningRate <= 0 || learningRate > 1) {
            throw RLException("Learning rate must be in range (0, 1]");
        }
        
        if (discountFactor < 0 || discountFactor >= 1) {
            throw RLException("Discount factor must be in range [0, 1)");
        }
        
        if (explorationRate < 0 || explorationRate > 1) {
            throw RLException("Exploration rate must be in range [0, 1]");
        }
        
        if (explorationDecay <= 0 || explorationDecay > 1) {
            throw RLException("Exploration decay must be in range (0, 1]");
        }
        
        if (minExplorationRate < 0 || minExplorationRate > explorationRate) {
            throw RLException("Min exploration rate must be in range [0, explorationRate]");
        }
    }

    /**
     * @brief Choose an action based on current state using epsilon-greedy policy
     * 
     * @param state The current state
     * @return Action The chosen action
     */
    Action chooseAction(const State& state) override {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Get available actions (assuming all actions are available)
        std::vector<Action> actions = ALL_ACTIONS;
        
        // Exploration: choose random action with probability epsilon
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        if (distribution(*randomGenerator) < explorationRate) {
            std::uniform_int_distribution<size_t> actionDist(0, actions.size() - 1);
            return actions[actionDist(*randomGenerator)];
        }
        
        // Exploitation: choose best action based on Q-values
        return getBestAction(state, actions);
    }

    /**
     * @brief Update Q-values based on experience using Q-learning update rule
     * 
     * Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
     * 
     * @param state Current state
     * @param action Taken action
     * @param reward Received reward
     * @param nextState Resulting state
     * @param done Whether episode is done
     */
    void update(
        const State& state,
        const Action& action,
        double reward,
        const State& nextState,
        bool done
    ) override {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Initialize Q-value for state-action pair if not already present
        if (qTable.find(state) == qTable.end() || 
            qTable[state].find(action) == qTable[state].end()) {
            qTable[state][action] = 0.0;
        }
        
        // Calculate maximum Q-value for next state
        double maxNextQ = 0.0;
        if (!done) { // If not terminal state
            maxNextQ = getMaxQValue(nextState);
        }
        
        // Current Q-value
        double currentQ = qTable[state][action];
        
        // Q-learning update rule
        double newQ = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
        qTable[state][action] = newQ;
        
        // Decay exploration rate
        explorationRate = std::max(
            minExplorationRate,
            explorationRate * explorationDecay
        );
    }

    /**
     * @brief Save the Q-table to a file
     * 
     * @param filename Path to save the file
     * @throws RLException if file cannot be opened
     */
    void savePolicy(const std::string& filename) const override {
        std::lock_guard<std::mutex> lock(mutex);
        
        try {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw RLException("Failed to open file for saving policy: " + filename);
            }
            
            // Write Q-table to file
            for (const auto& [state, actionValues] : qTable) {
                for (const auto& [action, value] : actionValues) {
                    file << state.x << "," << state.y << "," 
                         << static_cast<int>(action) << "," << value << "\n";
                }
            }
            
            file.close();
        } catch (const std::exception& e) {
            throw RLException(std::string("Error saving policy: ") + e.what());
        }
    }

    /**
     * @brief Load a Q-table from a file
     * 
     * @param filename Path to the file
     * @throws RLException if file cannot be opened or format is invalid
     */
    void loadPolicy(const std::string& filename) override {
        std::lock_guard<std::mutex> lock(mutex);
        
        try {
            // Check if file exists
            if (!std::filesystem::exists(filename)) {
                throw RLException("Policy file does not exist: " + filename);
            }
            
            std::ifstream file(filename);
            if (!file.is_open()) {
                throw RLException("Failed to open file for loading policy: " + filename);
            }
            
            // Clear existing Q-table
            qTable.clear();
            
            // Read Q-table from file
            std::string line;
            while (std::getline(file, line)) {
                int x, y, actionInt;
                double value;
                
                // Parse line
                std::sscanf(line.c_str(), "%d,%d,%d,%lf", &x, &y, &actionInt, &value);
                
                // Validate action
                if (actionInt < 0 || actionInt > 3) {
                    throw RLException("Invalid action in policy file: " + std::to_string(actionInt));
                }
                
                // Convert to state and action
                State state{x, y};
                Action action = static_cast<Action>(actionInt);
                
                // Update Q-table
                qTable[state][action] = value;
            }
            
            file.close();
        } catch (const std::exception& e) {
            throw RLException(std::string("Error loading policy: ") + e.what());
        }
    }

    /**
     * @brief Get the exploration rate
     * 
     * @return double The current exploration rate
     */
    double getExplorationRate() const {
        std::lock_guard<std::mutex> lock(mutex);
        return explorationRate;
    }
    
    /**
     * @brief Get the best action for a state based on current Q-values
     * 
     * @param state The state
     * @param actions Available actions
     * @return Action The best action
     */
    Action getBestAction(const State& state, const std::vector<Action>& actions) const {
        // If state not seen before, return random action
        if (qTable.find(state) == qTable.end()) {
            std::uniform_int_distribution<size_t> dist(0, actions.size() - 1);
            return actions[dist(*randomGenerator)];
        }
        
        const auto& stateQValues = qTable.at(state);
        
        // Find action with highest Q-value
        double maxQValue = std::numeric_limits<double>::lowest();
        std::vector<Action> bestActions;
        
        for (const auto& action : actions) {
            double qValue = 0.0; // Default Q-value for unseen actions
            
            // If action has been tried before, get its Q-value
            if (stateQValues.find(action) != stateQValues.end()) {
                qValue = stateQValues.at(action);
            }
            
            if (qValue > maxQValue) {
                maxQValue = qValue;
                bestActions.clear();
                bestActions.push_back(action);
            } else if (qValue == maxQValue) {
                bestActions.push_back(action);
            }
        }
        
        // If multiple actions have the same Q-value, select randomly among them
        if (bestActions.size() > 1) {
            std::uniform_int_distribution<size_t> dist(0, bestActions.size() - 1);
            return bestActions[dist(*randomGenerator)];
        }
        
        return bestActions[0];
    }

private:
    /**
     * @brief Get the maximum Q-value for all actions in a state
     * 
     * @param state The state
     * @return double The maximum Q-value
     */
    double getMaxQValue(const State& state) const {
        // If state not seen before, return 0
        if (qTable.find(state) == qTable.end()) {
            return 0.0;
        }
        
        const auto& stateQValues = qTable.at(state);
        
        // If no actions tried yet, return 0
        if (stateQValues.empty()) {
            return 0.0;
        }
        
        // Find maximum Q-value
        double maxQValue = std::numeric_limits<double>::lowest();
        for (const auto& [action, qValue] : stateQValues) {
            maxQValue = std::max(maxQValue, qValue);
        }
        
        return maxQValue;
    }
};

/**
 * @brief Train a Q-learning agent in a grid world environment
 * 
 * @param environment The grid world environment
 * @param agent The Q-learning agent
 * @param episodes Number of episodes to train
 * @param maxSteps Maximum steps per episode
 * @param verbose Whether to print training progress
 * @return Average reward per episode
 */
double trainAgent(
    std::shared_ptr<Environment> environment,
    std::shared_ptr<QLearningAgent> agent,
    int episodes,
    int maxSteps,
    bool verbose = false
) {
    // Record total reward for calculating average
    double totalReward = 0.0;
    
    // Training loop
    for (int episode = 0; episode < episodes; ++episode) {
        // Reset environment
        State state = environment->reset();
        double episodeReward = 0.0;
        int steps = 0;
        bool done = false;
        
        // Episode loop
        while (!done && steps < maxSteps) {
            // Choose action
            Action action = agent->chooseAction(state);
            
            // Take action
            auto [nextState, reward, isDone] = environment->step(action);
            
            // Update agent
            agent->update(state, action, reward, nextState, isDone);
            
            // Update state and accumulators
            state = nextState;
            episodeReward += reward;
            done = isDone;
            steps++;
        }
        
        totalReward += episodeReward;
        
        // Print progress
        if (verbose && episode % 100 == 0) {
            std::cout << "Episode: " << (episode + 1) << "/" << episodes 
                      << ", Steps: " << steps 
                      << ", Reward: " << std::fixed << std::setprecision(2) << episodeReward 
                      << ", Exploration rate: " << std::fixed << std::setprecision(4) << agent->getExplorationRate() 
                      << "\n";
        }
    }
    
    return totalReward / episodes;
}

/**
 * @brief Test a trained agent in the environment
 * 
 * @param environment The environment
 * @param agent The trained agent
 * @param episodes Number of test episodes
 * @param maxSteps Maximum steps per episode
 * @param render Whether to visualize the environment
 * @return Average reward per episode
 */
double testAgent(
    std::shared_ptr<Environment> environment,
    std::shared_ptr<QLearningAgent> agent,
    int episodes,
    int maxSteps,
    bool render = true
) {
    double totalReward = 0.0;
    int totalSteps = 0;
    int successCount = 0;
    
    for (int episode = 0; episode < episodes; ++episode) {
        State state = environment->reset();
        double episodeReward = 0.0;
        int steps = 0;
        bool done = false;
        
        if (render) {
            std::cout << "\nTest Episode " << (episode + 1) << " Start:\n";
            environment->render();
        }
        
        while (!done && steps < maxSteps) {
            // In testing, always choose best action
            Action action = agent->getBestAction(state, ALL_ACTIONS);
            
            // Take action
            auto [nextState, reward, isDone] = environment->step(action);
            
            // Update state and accumulators
            state = nextState;
            episodeReward += reward;
            done = isDone;
            steps++;
            
            if (render) {
                std::cout << "Step " << steps << ": Action = " << actionToString(action) 
                          << ", Reward = " << std::fixed << std::setprecision(2) << reward << "\n";
                environment->render();
                
                // Add small delay for visualization
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }
        }
        
        totalReward += episodeReward;
        totalSteps += steps;
        if (done) successCount++;
        
        if (render) {
            std::cout << "Episode " << (episode + 1) << " finished. Steps: " << steps 
                      << ", Reward: " << std::fixed << std::setprecision(2) << episodeReward 
                      << ", Success: " << (done ? "Yes" : "No") << "\n";
        }
    }
    
    double avgReward = totalReward / episodes;
    double avgSteps = static_cast<double>(totalSteps) / episodes;
    double successRate = static_cast<double>(successCount) / episodes * 100.0;
    
    std::cout << "\nTest Results:\n";
    std::cout << "Average Reward: " << std::fixed << std::setprecision(2) << avgReward << "\n";
    std::cout << "Average Steps: " << std::fixed << std::setprecision(2) << avgSteps << "\n";
    std::cout << "Success Rate: " << std::fixed << std::setprecision(2) << successRate << "%\n";
    
    return avgReward;
}

/**
 * @brief Create a simple grid world environment
 * 
 * @return Shared pointer to the environment
 */
std::shared_ptr<Environment> createSimpleGridWorld() {
    // Grid parameters
    int width = 5;
    int height = 5;
    
    // Initial and goal states
    State initialState{0, 0};
    State goalState{4, 4};
    
    // Obstacles
    std::vector<State> obstacles = {
        {1, 1}, {1, 2}, {1, 3},
        {3, 0}, {3, 1}, {3, 2}
    };
    
    // Create environment
    return std::make_shared<GridWorld>(
        width, height, initialState, goalState, obstacles
    );
}

/**
 * @brief Run a complete training and testing cycle
 */
void runReinforcementLearningDemo() {
    try {
        // Create environment
        auto environment = createSimpleGridWorld();
        
        // Create agent
        auto agent = std::make_shared<QLearningAgent>(
            0.1,    // learning rate
            0.99,   // discount factor
            1.0,    // initial exploration rate
            0.995,  // exploration decay
            0.01    // minimum exploration rate
        );
        
        // Training parameters
        int trainingEpisodes = 1000;
        int maxSteps = 100;
        bool verbose = true;
        
        std::cout << "Starting training...\n";
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Train agent
        double avgReward = trainAgent(
            environment, agent, trainingEpisodes, maxSteps, verbose
        );
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime
        ).count();
        
        std::cout << "Training completed in " 
                  << std::fixed << std::setprecision(2) << (duration / 1000.0) 
                  << " seconds.\n";
        std::cout << "Average reward: " << std::fixed << std::setprecision(2) << avgReward << "\n";
        
        // Optional: Save policy
        try {
            agent->savePolicy("q_learning_policy.txt");
            std::cout << "Policy saved to q_learning_policy.txt\n";
        } catch (const RLException& e) {
            std::cerr << "Warning: " << e.what() << '\n';
        }
        
        // Test agent
        std::cout << "\nTesting trained agent...\n";
        testAgent(environment, agent, 5, maxSteps, true);
        
    } catch (const RLException& e) {
        std::cerr << "Error: " << e.what() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << '\n';
    }
}

/**
 * @brief Main function
 */
int main() {
    std::cout << "Reinforcement Learning Demo: Q-Learning in Grid World\n";
    runReinforcementLearningDemo();
    return 0;
}