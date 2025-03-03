/**
 * @file ThreadPool.hpp
 * @brief Thread Pool -- pre-creates worker threads that process a shared task queue.
 *
 * @details
 * A fixed number of worker threads wait on a condition variable for tasks.
 * Avoids the overhead of creating/destroying threads for each unit of work.
 *
 * C++20 modernization:
 *   - std::jthread replaces std::thread for automatic joining on destruction.
 *   - std::stop_token provides cooperative cancellation, eliminating the manual
 *     bool shutdown flag entirely.
 *   - std::condition_variable_any works with stop_token natively, allowing the
 *     wait to return immediately when stop is requested.
 *
 * How it works:
 *   1. Constructor spawns N jthread workers.
 *   2. Each worker loops: wait for a task or stop signal -> pop from queue -> execute.
 *   3. Enqueue() pushes a task and wakes one idle worker.
 *   4. EnqueueWithFuture() returns a std::future for the result.
 *   5. Destructor: jthread's destructor calls request_stop() then join()
 *      automatically -- no manual flag or join loop needed.
 *
 * Thread-safety:
 *   - A single mutex protects the task queue.
 *   - A condition_variable_any wakes idle workers when tasks arrive.
 *   - Stop is propagated through the stop_token mechanism (lock-free).
 *
 * Complexity:
 *   - Enqueue():  O(1)
 *   - Shutdown:   O(remaining_tasks + num_threads)
 */

#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <future>
#include <stop_token>


/**
 * @class ThreadPool
 * @brief Fixed-size pool of jthread workers processing a shared task queue.
 *
 * Uses C++20 std::jthread + std::stop_token for cooperative cancellation,
 * removing the need for a manual shutdown flag.
 */
class ThreadPool {
public:
    /**
     * @brief Create a pool with the given number of worker threads.
     * @param num_threads Number of workers (defaults to hardware concurrency).
     */
    explicit ThreadPool(std::size_t num_threads = std::jthread::hardware_concurrency());

    /**
     * @brief Destructor -- jthread's destructor requests stop and joins automatically.
     *
     * Any tasks still in the queue are executed before workers exit.
     */
    ~ThreadPool();

    /** @brief Non-copyable, non-movable. */
    ThreadPool(const ThreadPool&) = delete;
    /** @brief Non-copyable, non-movable. */
    ThreadPool& operator=(const ThreadPool&) = delete;

    /**
     * @brief Add a fire-and-forget task to the queue.
     * @param task Callable to execute.
     * @throws std::runtime_error if the pool has been stopped.
     */
    void Enqueue(std::function<void()> task);

    /**
     * @brief Add a task and get a std::future for its return value.
     * @tparam F Callable type.
     * @tparam Args Argument types.
     * @param func Callable to execute.
     * @param args Arguments to forward to the callable.
     * @return std::future for the result.
     * @throws std::runtime_error if the pool has been stopped.
     */
    template <typename F, typename... Args>
    auto EnqueueWithFuture(F&& func, Args&&... args) {
        // decltype(func(args...)) = "the type you get when you call func(args...)"
        using ReturnType = decltype(func(args...));

        auto task_ptr = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(func), std::forward<Args>(args)...)
        );

        std::future<ReturnType> future = task_ptr->get_future();
        Enqueue([task_ptr]() { (*task_ptr)(); });
        return future;
    }

    /**
     * @brief Number of worker threads.
     * @return Thread count.
     */
    std::size_t ThreadCount() const;

    /**
     * @brief Number of tasks waiting in the queue.
     * @return Pending task count.
     */
    std::size_t PendingTasks() const;

private:
    /**
     * @brief Main loop every worker thread runs.
     * @param stop_token Cooperative cancellation token from jthread.
     *
     * Waits for a task (or the stop signal), then executes it.
     * When stop is requested, drains remaining tasks before returning.
     */
    void WorkerLoop(std::stop_token stop_token);

    std::queue<std::function<void()>>    task_queue_;   ///< Pending task queue.
    mutable std::mutex                   queue_mutex_;  ///< Queue guard.
    std::condition_variable_any          cv_;           ///< Worker wake signal.
    std::vector<std::jthread>            workers_;      ///< Worker jthreads (last: destroyed first).
};

#endif // THREAD_POOL_HPP
