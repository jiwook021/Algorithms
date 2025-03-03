#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <random>
#include <condition_variable>

// Number of philosophers
const int NUM_PHILOSOPHERS = 5;

// State of each philosopher
enum class State { THINKING, HUNGRY, EATING };

// Mutex for protecting access to shared data
std::mutex mtx;

// Array to store state of each philosopher
std::vector<State> state(NUM_PHILOSOPHERS, State::THINKING);

// Condition variables for each philosopher to wait on
std::vector<std::condition_variable> cv(NUM_PHILOSOPHERS);

// Mutex for each fork
std::vector<std::mutex> forks(NUM_PHILOSOPHERS);

// Random number generator for thinking and eating times
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> think_dist(1000, 3000);  // 1-3 seconds
std::uniform_int_distribution<> eat_dist(1000, 2000);    // 1-2 seconds

// Function to check if a philosopher can eat
bool can_eat(int id) {
    int left = (id + NUM_PHILOSOPHERS - 1) % NUM_PHILOSOPHERS;
    int right = (id + 1) % NUM_PHILOSOPHERS;
    return (state[id] == State::HUNGRY &&
            state[left] != State::EATING &&
            state[right] != State::EATING);
}

// Function to signal a philosopher that they can eat
void test(int id) {
    if (can_eat(id)) {
        state[id] = State::EATING;
        cv[id].notify_one();
    }
}

// Function for a philosopher to pick up forks
void pickup_forks(int id) {
    std::unique_lock<std::mutex> lock(mtx);
    state[id] = State::HUNGRY;
    std::cout << "Philosopher " << id << " is hungry" << std::endl;
    
    // Try to acquire forks
    test(id);
    
    // Wait if cannot eat
    while (state[id] != State::EATING) {
        cv[id].wait(lock);
    }
    
    std::cout << "Philosopher " << id << " is eating" << std::endl;
}

// Function for a philosopher to put down forks
void putdown_forks(int id) {
    std::unique_lock<std::mutex> lock(mtx);
    state[id] = State::THINKING;
    std::cout << "Philosopher " << id << " is thinking" << std::endl;
    
    // Test if neighbors can eat
    int left = (id + NUM_PHILOSOPHERS - 1) % NUM_PHILOSOPHERS;
    int right = (id + 1) % NUM_PHILOSOPHERS;
    
    test(left);
    test(right);
}

// Function for philosopher behavior
void philosopher(int id) {
    while (true) {
        // Think for a while
        std::cout << "Philosopher " << id << " is thinking" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(think_dist(gen)));
        
        // Pick up forks
        pickup_forks(id);
        
        // Eat for a while
        std::this_thread::sleep_for(std::chrono::milliseconds(eat_dist(gen)));
        
        // Put down forks
        putdown_forks(id);
    }
}

int main() {
    std::vector<std::thread> philosophers;
    
    std::cout << "Dining philosophers problem simulation" << std::endl;
    std::cout << "Press Ctrl+C to exit" << std::endl;
    
    // Create philosopher threads
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        philosophers.push_back(std::thread(philosopher, i));
    }
    
    // Join threads (though they never terminate in this implementation)
    for (auto& p : philosophers) {
        p.join();
    }
    
    return 0;
}