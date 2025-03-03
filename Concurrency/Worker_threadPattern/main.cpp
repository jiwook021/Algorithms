#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <atomic>
#include <iostream>

// WorkerThreadPool class encapsulates thread pool management and task execution
class WorkerThreadPool {
public:
    // Constructor initializes worker threads based on the specified thread count
    explicit WorkerThreadPool(size_t threadCount)
        : isRunning_(true)
    {
        // Creating worker threads
        for (size_t i = 0; i < threadCount; ++i) {
            workers_.emplace_back([this] { workerThread(); });
        }
    }

    // Destructor ensures proper shutdown of worker threads
    ~WorkerThreadPool() {
        shutdown();
    }

    // Enqueue a new task to be executed by worker threads
    void enqueueTask(const std::function<void()>& task) {
        {
            // Lock mutex to ensure thread-safe access to task queue
            std::lock_guard<std::mutex> lock(queueMutex_);
            tasks_.push(task);
        }
        // Notify one worker thread that a new task is available
        condition_.notify_one();
    }

    // Graceful shutdown, waits for all threads to finish
    void shutdown() {
        // Indicate shutdown to worker threads
        isRunning_ = false;
        condition_.notify_all(); // Wake up all worker threads

        // Join all threads to main thread
        for (auto& thread : workers_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

private:
    std::vector<std::thread> workers_; // Vector storing worker threads
    std::queue<std::function<void()>> tasks_; // Task queue
    std::mutex queueMutex_; // Mutex for protecting task queue access
    std::condition_variable condition_; // Condition variable for synchronization
    std::atomic<bool> isRunning_; // Atomic flag for running state

    // Function executed by each worker thread to process tasks
    void workerThread() {
        while (isRunning_) {
            std::function<void()> task;
            {
                // Wait for a task or shutdown signal
                std::unique_lock<std::mutex> lock(queueMutex_);
                condition_.wait(lock, [this] {
                    return !tasks_.empty() || !isRunning_;
                });

                // Check shutdown condition
                if (!isRunning_ && tasks_.empty()) {
                    return;
                }

                // Retrieve next task from queue
                task = tasks_.front();
                tasks_.pop();
            }

            // Execute task outside the lock to avoid blocking queue operations
            try {
                task();
            } catch (const std::exception& e) {
                // Handle exceptions thrown by task
                std::cerr << "Exception caught during task execution: " << e.what() << std::endl;
            }
        }
    }
};

// Usage example demonstrating WorkerThreadPool in action
int main() {
    WorkerThreadPool pool(4); // Initialize worker pool with 4 threads

    // Enqueue sample tasks
    for (int i = 0; i < 10; ++i) {
        pool.enqueueTask([i] {
            std::cout << "Processing task #" << i << " on thread ID: "
                      << std::this_thread::get_id() << "\n";
        });
    }

    // Allow some time for tasks to complete
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Shutdown the pool gracefully
    pool.shutdown();

    return 0;
}
