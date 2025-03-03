#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

class ThreadPool {
private:
    // Collection of worker threads
    std::vector<std::thread> workers;
    // Queue of tasks to be executed
    std::queue<std::function<void()>> tasks;
    
    // Synchronization objects
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    // Constructor: initialize the thread pool with the specified number of threads
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) : stop(false) {
        // Create the specified number of worker threads
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                // Each thread executes this lambda function
                while (true) {
                    std::function<void()> task;
                    
                    // Get a task from the queue
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        
                        // Wait until there is a task or the pool is stopping
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        
                        // If the pool is stopping and there are no more tasks, exit
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        
                        // Get the task from the front of the queue
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    
                    // Execute the task
                    task();
                }
            });
        }
    }
    
    // Add a new task to the thread pool
    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // Don't allow enqueueing after stopping the pool
            if (stop) {
                throw std::runtime_error("Cannot add task to stopped ThreadPool");
            }
            
            // Add the task to the queue
            tasks.push(task);
        }
        
        // Notify one waiting thread that there's a new task
        condition.notify_one();
    }
    
    // Destructor: clean up all threads
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        
        // Wake up all threads so they can finish and exit
        condition.notify_all();
        
        // Wait for all threads to finish
        for (std::thread &worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
};

// Example usage
int main() {
    // Create a thread pool with 4 worker threads
    ThreadPool pool(4);
    
    // Enqueue some tasks
    for (int i = 0; i < 8; ++i) {
        pool.enqueue([i] {
            // Each task prints its ID and the thread it's running on
            std::cout << "Task " << i << " executed by thread ID: " 
                      << std::this_thread::get_id() << std::endl;
            
            // Simulate work with a sleep
            std::this_thread::sleep_for(std::chrono::seconds(1));
        });
    }
    
    // Give some time for tasks to complete
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    std::cout << "Main thread: All tasks have been submitted." << std::endl;
    
    // The pool will be destroyed when it goes out of scope,
    // which will complete all remaining tasks and join all threads
    return 0;
}