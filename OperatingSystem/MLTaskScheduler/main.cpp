// MLTaskScheduler.cpp - A Machine Learning-based Task Scheduler in C++17
// This scheduler uses a simple ML model to predict task execution times and
// optimally schedules tasks based on these predictions and system constraints.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <stdexcept>
#include <memory>
#include <iomanip>

// Global mutex for synchronizing console output
std::mutex g_cout_mutex;

// Helper function for synchronized console output
template<typename... Args>
void synchronized_cout(Args&&... args) {
    std::lock_guard<std::mutex> lock(g_cout_mutex);
    (std::cout << ... << std::forward<Args>(args));
}

// Constants for the ML model and scheduler
constexpr size_t DEFAULT_THREAD_POOL_SIZE = 4;
constexpr int MAX_PRIORITY = 10;
constexpr int DEFAULT_PRIORITY = 5;
constexpr double DEFAULT_LEARNING_RATE = 0.01;

// Simple linear regression model for task execution time prediction
class MLModel {
public:
    // Constructor initializes weights randomly
    MLModel(size_t feature_count = 3, double learning_rate = DEFAULT_LEARNING_RATE)
        : weights_(feature_count, 0.0), 
          bias_(0.0), 
          learning_rate_(learning_rate) {
        
        // Initialize weights with small random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.1, 0.1);
        
        for (auto& weight : weights_) {
            weight = dis(gen);
        }
    }
    
    // Predict task execution time based on features
    // Time Complexity: O(n) where n is number of features
    // Space Complexity: O(1)
    double predict(const std::vector<double>& features) const {
        if (features.size() != weights_.size()) {
            throw std::invalid_argument("Feature count doesn't match model weight count");
        }
        
        // Linear regression formula: y = w1*x1 + w2*x2 + ... + wn*xn + b
        double prediction = bias_;
        for (size_t i = 0; i < features.size(); i++) {
            prediction += weights_[i] * features[i];
        }
        
        // Ensure prediction is positive (execution time can't be negative)
        return std::max(0.0, prediction);
    }
    
    // Train the model with a new observation
    // Time Complexity: O(n) where n is number of features
    // Space Complexity: O(1)
    void train(const std::vector<double>& features, double actual_time) {
        if (features.size() != weights_.size()) {
            throw std::invalid_argument("Feature count doesn't match model weight count");
        }
        
        // Predict with current weights
        double prediction = predict(features);
        
        // Compute error
        double error = actual_time - prediction;
        
        // Update weights using gradient descent
        for (size_t i = 0; i < weights_.size(); i++) {
            weights_[i] += learning_rate_ * error * features[i];
        }
        
        // Update bias
        bias_ += learning_rate_ * error;

        // Log the model update for debugging
        std::lock_guard<std::mutex> lock(g_cout_mutex);
        std::cout << "Model updated - Prediction: " << std::fixed << std::setprecision(2) 
                  << prediction << "ms, Actual: " << actual_time 
                  << "ms, Error: " << error << "ms" << std::endl;
        std::cout << "New weights: [";
        for (size_t i = 0; i < weights_.size(); i++) {
            std::cout << std::fixed << std::setprecision(3) << weights_[i];
            if (i < weights_.size() - 1) std::cout << ", ";
        }
        std::cout << "], Bias: " << bias_ << std::endl;
    }
    
    // Get the current model weights (for debugging/logging)
    std::vector<double> getWeights() const {
        return weights_;
    }
    
    double getBias() const {
        return bias_;
    }
    
private:
    std::vector<double> weights_;  // Model weights
    double bias_;                  // Model bias term
    double learning_rate_;         // Learning rate for training
    mutable std::mutex model_mutex_; // Protects model state during concurrent access
};

// Task class representing a unit of work to be scheduled
class Task {
public:
    using TaskFunction = std::function<void()>;
    
    // Constructor
    Task(TaskFunction func, 
         std::string name, 
         int priority = DEFAULT_PRIORITY,
         std::vector<double> features = {})
        : func_(std::move(func)),
          name_(std::move(name)),
          priority_(std::clamp(priority, 1, MAX_PRIORITY)),
          creation_time_(std::chrono::steady_clock::now()),
          features_(std::move(features)),
          predicted_duration_(0.0),
          actual_duration_(0.0) {
        
        // Validate task function
        if (!func_) {
            throw std::invalid_argument("Task function cannot be null");
        }
        
        // Validate task name
        if (name_.empty()) {
            throw std::invalid_argument("Task name cannot be empty");
        }
    }
    
    // Execute the task and measure its execution time
    void execute() {
        auto start = std::chrono::steady_clock::now();
        
        try {
            func_();
        } catch (const std::exception& e) {
            synchronized_cout("Task '", name_, "' failed with exception: ", e.what(), "\n");
        } catch (...) {
            synchronized_cout("Task '", name_, "' failed with unknown exception\n");
        }
        
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = end - start;
        actual_duration_ = duration.count();
    }
    
    // Getters
    const std::string& getName() const { return name_; }
    int getPriority() const { return priority_; }
    std::chrono::steady_clock::time_point getCreationTime() const { return creation_time_; }
    const std::vector<double>& getFeatures() const { return features_; }
    double getPredictedDuration() const { return predicted_duration_; }
    double getActualDuration() const { return actual_duration_; }
    
    // Setters
    void setPredictedDuration(double duration) { predicted_duration_ = duration; }
    
private:
    TaskFunction func_;                              // The function to execute
    std::string name_;                               // Task name for identification
    int priority_;                                   // Priority level (1-10, higher = more important)
    std::chrono::steady_clock::time_point creation_time_;  // When the task was created
    std::vector<double> features_;                   // Features for ML prediction (e.g., data size, complexity)
    double predicted_duration_;                      // Predicted execution time by ML model
    double actual_duration_;                         // Actual execution time (set after execution)
};

// Custom comparator for task priority queue
struct TaskComparator {
    // Time Complexity: O(1)
    // Space Complexity: O(1)
    bool operator()(const std::shared_ptr<Task>& a, const std::shared_ptr<Task>& b) const {
        // Higher priority tasks come first (note: in priority_queue, we use greater-than for earlier placement)
        if (a->getPriority() != b->getPriority()) {
            return a->getPriority() < b->getPriority();
        }
        
        // If priorities are equal, shorter predicted tasks come first
        if (a->getPredictedDuration() != b->getPredictedDuration()) {
            return a->getPredictedDuration() > b->getPredictedDuration();
        }
        
        // If predicted durations are equal, older tasks come first (FIFO)
        return a->getCreationTime() > b->getCreationTime();
    }
};

// The main ML-based task scheduler class
class TaskScheduler {
public:
    // Constructor
    explicit TaskScheduler(size_t thread_count = DEFAULT_THREAD_POOL_SIZE) 
        : thread_count_(thread_count),
          running_(true) {
        
        // Input validation
        if (thread_count_ == 0) {
            throw std::invalid_argument("Thread count must be at least 1");
        }
        
        // Initialize ML model
        model_ = std::make_unique<MLModel>();
        
        // Initialize statistics
        stats_["total_tasks_executed"] = 0.0;
        stats_["total_prediction_error"] = 0.0;
        stats_["avg_prediction_error"] = 0.0;
        stats_["total_execution_time"] = 0.0;
        stats_["avg_execution_time"] = 0.0;
        
        // Start worker threads
        for (size_t i = 0; i < thread_count_; i++) {
            threads_.emplace_back([this, i]() {
                this->workerThread(i);
            });
        }
    }
    
    // Destructor - ensure clean shutdown
    ~TaskScheduler() {
        shutdown();
    }
    
    // Prevent copying or moving
    TaskScheduler(const TaskScheduler&) = delete;
    TaskScheduler& operator=(const TaskScheduler&) = delete;
    TaskScheduler(TaskScheduler&&) = delete;
    TaskScheduler& operator=(TaskScheduler&&) = delete;
    
    // Submit a task to the scheduler
    // Time Complexity: O(log n) for priority queue insertion
    // Space Complexity: O(1)
    std::future<void> submitTask(Task::TaskFunction func, 
                                 const std::string& name, 
                                 int priority = DEFAULT_PRIORITY,
                                 const std::vector<double>& features = {}) {
        // Input validation
        if (!func) {
            throw std::invalid_argument("Task function cannot be null");
        }
        
        if (name.empty()) {
            throw std::invalid_argument("Task name cannot be empty");
        }
        
        if (priority < 1 || priority > MAX_PRIORITY) {
            throw std::invalid_argument("Priority must be between 1 and " + 
                                         std::to_string(MAX_PRIORITY));
        }
        
        // Create a promise/future pair to track task completion
        std::shared_ptr<std::promise<void>> promise = std::make_shared<std::promise<void>>();
        std::future<void> future = promise->get_future();
        
        // Wrap the task function to fulfill the promise when done
        auto wrapped_func = [func = std::move(func), promise]() {
            try {
                func();
                promise->set_value();
            } catch (...) {
                try {
                    promise->set_exception(std::current_exception());
                } catch (...) {
                    // Handle broken promise scenario
                }
            }
        };
        
        // Create the task object
        auto task = std::make_shared<Task>(std::move(wrapped_func), name, priority, features);
        
        // Use ML model to predict execution time
        if (!features.empty()) {
            double predicted_time = model_->predict(features);
            task->setPredictedDuration(predicted_time);
            
            synchronized_cout("Task '", name, "' predicted duration: ", 
                              std::fixed, std::setprecision(2), predicted_time, "ms\n");
        }
        
        // Add task to the queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            // This lock is needed to protect the task queue from concurrent
            // modifications by multiple threads submitting tasks simultaneously
            task_queue_.push(task);
        }
        
        // Notify one worker thread that a new task is available
        condition_.notify_one();
        
        return future;
    }
    
    // Shutdown the scheduler gracefully
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            // This lock ensures that we don't have race conditions during shutdown
            if (!running_) {
                return;  // Already shutdown
            }
            running_ = false;
        }
        
        // Notify all threads to check the running_ flag
        condition_.notify_all();
        
        // Wait for all threads to finish
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        threads_.clear();
    }
    
    // Get statistics about the scheduler
    std::map<std::string, double> getStats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        // Lock needed to provide consistent view of stats
        return stats_;
    }
    
    // Get current queue size
    size_t getQueueSize() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return task_queue_.size();
    }
    
private:
    // Worker thread function
    void workerThread(size_t thread_id) {
        while (true) {
            std::shared_ptr<Task> task;
            
            // Wait for a task or shutdown signal
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                // Lock required to protect shared condition variable and task queue
                condition_.wait(lock, [this]() {
                    return !running_ || !task_queue_.empty();
                });
                
                // Check if we should exit
                if (!running_ && task_queue_.empty()) {
                    break;
                }
                
                // Get next task from queue
                if (!task_queue_.empty()) {
                    task = task_queue_.top();
                    task_queue_.pop();
                }
            }
            
            // Execute the task if we got one
            if (task) {
                try {
                    synchronized_cout("Thread ", thread_id, " executing task: ", 
                                     task->getName(), "\n");
                    
                    // Execute the task and measure actual time
                    task->execute();
                    
                    // Update the ML model with the actual execution time
                    if (!task->getFeatures().empty()) {
                        model_->train(task->getFeatures(), task->getActualDuration());
                    }
                    
                    // Update statistics
                    updateStats(task);
                    
                } catch (const std::exception& e) {
                    synchronized_cout("Error in worker thread ", thread_id, ": ", 
                                     e.what(), "\n");
                } catch (...) {
                    synchronized_cout("Unknown error in worker thread ", thread_id, "\n");
                }
            }
        }
        
        synchronized_cout("Worker thread ", thread_id, " exiting\n");
    }
    
    // Update scheduler statistics
    void updateStats(const std::shared_ptr<Task>& task) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        // Lock needed for thread-safe statistics update
        
        // Track total tasks executed
        stats_["total_tasks_executed"] += 1.0;
        
        // Track average prediction error
        double prediction_error = std::abs(task->getActualDuration() - task->getPredictedDuration());
        stats_["total_prediction_error"] += prediction_error;
        stats_["avg_prediction_error"] = stats_["total_prediction_error"] / stats_["total_tasks_executed"];
        
        // Track average execution time
        stats_["total_execution_time"] += task->getActualDuration();
        stats_["avg_execution_time"] = stats_["total_execution_time"] / stats_["total_tasks_executed"];
    }
    
    // Member variables
    size_t thread_count_;                       // Number of worker threads
    std::atomic<bool> running_;                 // Flag to control thread execution
    std::vector<std::thread> threads_;          // Worker threads
    
    // Task queue with priority-based comparison
    std::priority_queue<
        std::shared_ptr<Task>,
        std::vector<std::shared_ptr<Task>>,
        TaskComparator
    > task_queue_;
    
    mutable std::mutex queue_mutex_;            // Protects task_queue_ (mutable for const methods)
    std::condition_variable condition_;         // For worker thread synchronization
    
    std::unique_ptr<MLModel> model_;            // ML model for prediction
    
    mutable std::mutex stats_mutex_;            // Protects stats_
    std::map<std::string, double> stats_;       // Scheduler statistics
};

// Example usage and test code
int main() {
    try {
        // Create a scheduler with 4 worker threads
        TaskScheduler scheduler(4);
        
        // Vector to store futures for tracking task completion
        std::vector<std::future<void>> futures;
        
        // Random number generator for adding some variance to tasks
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> variance(0.8, 1.2);
        
        // Submit some tasks with different features
        for (int i = 0; i < 20; i++) {
            // Example features: data size, complexity, I/O operations
            std::vector<double> features = {
                static_cast<double>((i % 10) + 1),       // data size (1-10)
                static_cast<double>((i % 3) + 1),        // complexity (1-3)
                static_cast<double>(i > 10 ? 5.0 : 1.0)  // I/O operations (1 or 5)
            };
            
            // Create task function with more varied workload
            auto task_func = [i, features, &gen, &variance]() {
                // Create more diverse workloads that correlate with features
                double base_workload = features[0] * 20.0 + features[1] * 100.0 + features[2] * 30.0;
                
                // Add some randomness (+/- 20%)
                double workload = base_workload * variance(gen);
                
                // Simulate work
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(workload)));
                
                // Report completion with synchronized output
                synchronized_cout("Task ", i, " completed with workload: ", 
                                 std::fixed, std::setprecision(2), workload, "ms\n");
            };
            
            // Calculate priority (for demo purposes)
            int priority = (i % MAX_PRIORITY) + 1;
            
            // Submit the task
            std::string task_name = "Task_" + std::to_string(i);
            auto future = scheduler.submitTask(task_func, task_name, priority, features);
            futures.push_back(std::move(future));
        }
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        
        // Display scheduler statistics
        auto stats = scheduler.getStats();
        synchronized_cout("\nScheduler Statistics:\n");
        for (const auto& [key, value] : stats) {
            synchronized_cout(key, ": ", std::fixed, std::setprecision(5), value, "\n");
        }
        
        // Submit a second batch to see improved predictions
        synchronized_cout("\n=== Second batch of tasks with same feature patterns ===\n\n");
        
        futures.clear();
        
        // Submit more tasks with the same feature patterns to test learning
        for (int i = 0; i < 10; i++) {
            // Use the same feature patterns as some of the first batch
            int pattern_index = i * 2; // Use even indices from first batch
            
            std::vector<double> features = {
                static_cast<double>((pattern_index % 10) + 1),    
                static_cast<double>((pattern_index % 3) + 1),     
                static_cast<double>(pattern_index > 10 ? 5.0 : 1.0)
            };
            
            auto task_func = [i, features, &gen, &variance]() {
                // Same workload calculation as before
                double base_workload = features[0] * 20.0 + features[1] * 100.0 + features[2] * 30.0;
                double workload = base_workload * variance(gen);
                
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(workload)));
                
                synchronized_cout("Batch 2 - Task ", i, " completed with workload: ", 
                                 std::fixed, std::setprecision(2), workload, "ms\n");
            };
            
            std::string task_name = "Batch2_Task_" + std::to_string(i);
            auto future = scheduler.submitTask(task_func, task_name, DEFAULT_PRIORITY, features);
            futures.push_back(std::move(future));
        }
        
        // Wait for second batch to complete
        for (auto& future : futures) {
            future.wait();
        }
        
        // Display final statistics
        stats = scheduler.getStats();
        synchronized_cout("\nFinal Scheduler Statistics:\n");
        for (const auto& [key, value] : stats) {
            synchronized_cout(key, ": ", std::fixed, std::setprecision(5), value, "\n");
        }
        
        // Shutdown the scheduler
        scheduler.shutdown();
        
    } catch (const std::exception& e) {
        std::cerr << "Error in main: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error in main" << std::endl;
        return 1;
    }
    
    return 0;
}