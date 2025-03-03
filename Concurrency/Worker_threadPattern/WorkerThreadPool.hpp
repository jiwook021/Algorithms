/**
 * @file WorkerThreadPool.hpp
 * @brief Worker Thread Pattern -- a thread pool using std::jthread + stop_token.
 *
 * @details
 * Implements the Worker Thread pattern with C++20 cooperative cancellation:
 *   - std::jthread for automatic join and stop_token support.
 *   - std::condition_variable_any::wait() with stop_token for cooperative shutdown.
 *   - Explicit Shutdown() method requests stop on all workers.
 *   - Exception-safe task execution (catches and logs exceptions).
 *
 * Synchronization:
 *   - std::mutex + std::condition_variable_any guard the task queue.
 *   - std::stop_token enables cooperative cancellation without a separate
 *     atomic flag -- the jthread destructor requests stop automatically.
 *
 * Complexity:
 *   - EnqueueTask(): O(1)
 *   - Shutdown:      O(remaining_tasks + num_threads)
 */

#ifndef WORKER_THREAD_POOL_HPP
#define WORKER_THREAD_POOL_HPP

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <iostream>
#include <cstddef>

/**
 * @class WorkerThreadPool
 * @brief Thread pool with jthread-based cooperative shutdown and exception-safe
 *        task execution.
 */
class WorkerThreadPool {
public:
    /**
     * @brief Construct a pool with the given number of worker threads.
     * @param num_threads Number of workers to spawn.
     */
    explicit WorkerThreadPool(std::size_t num_threads);

    /** @brief Destructor calls Shutdown() to ensure all threads are stopped. */
    ~WorkerThreadPool();

    /** @brief Non-copyable. */
    WorkerThreadPool(const WorkerThreadPool&) = delete;
    /** @brief Non-copyable. */
    WorkerThreadPool& operator=(const WorkerThreadPool&) = delete;

    /**
     * @brief Enqueue a task for execution by a worker thread.
     * @param task Callable to execute.
     */
    void EnqueueTask(std::function<void()> task);

    /**
     * @brief Gracefully stop all workers by requesting stop on every jthread.
     *
     * Workers drain remaining tasks before exiting. Idempotent -- safe to call
     * multiple times.
     */
    void Shutdown();

    /**
     * @brief Return the number of worker threads.
     * @return Thread count.
     */
    std::size_t ThreadCount() const;

    /**
     * @brief Check if the pool is still accepting tasks.
     * @return True if running (no stop has been requested).
     */
    bool IsRunning() const;

    /**
     * @brief Return the approximate number of pending tasks.
     * @return Pending task count.
     */
    std::size_t PendingTasks() const;

private:
    /**
     * @brief Worker loop: wait for tasks via stop_token-aware condition
     *        variable, execute them, handle exceptions.
     * @param stop_token Cooperative cancellation token from std::jthread.
     */
    void WorkerLoop(std::stop_token stop_token);

    std::vector<std::jthread> workers_;              ///< Worker threads.
    std::queue<std::function<void()>> task_queue_;    ///< Pending task queue.
    mutable std::mutex queue_mutex_;                  ///< Queue guard.
    std::condition_variable_any cv_;                  ///< Worker wake signal.
};

#endif // WORKER_THREAD_POOL_HPP
