/**
 * @file ThreadPool.cpp
 * @brief Implementation of the ThreadPool class.
 *
 * @details
 * Separates non-template member definitions from the header.
 * Template methods (EnqueueWithFuture) remain in the header.
 *
 * Key C++20 feature: WorkerLoop receives a std::stop_token from jthread.
 * The condition_variable_any::wait() overload accepts a stop_token,
 * returning immediately when stop is requested -- no manual bool flag needed.
 */

#include "ThreadPool.hpp"

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

ThreadPool::ThreadPool(std::size_t num_threads) {
    workers_.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this](std::stop_token st) { WorkerLoop(st); });
    }
}

ThreadPool::~ThreadPool() {
    // Notify all workers so they can observe the stop request.
    // jthread's destructor will call request_stop() and join() automatically,
    // but we notify first to unblock any workers waiting on the CV.
    for (auto& worker : workers_) {
        worker.request_stop();
    }
    cv_.notify_all();
    // jthread destructors handle join automatically.
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void ThreadPool::Enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        // Check if any worker can still accept tasks.
        // If all workers have been stopped, reject the task.
        if (!workers_.empty() && workers_.front().get_stop_token().stop_requested()) {
            throw std::runtime_error("Cannot enqueue -- ThreadPool is stopped");
        }
        task_queue_.push(std::move(task));
    }
    cv_.notify_one();
}

std::size_t ThreadPool::ThreadCount() const {
    return workers_.size();
}

std::size_t ThreadPool::PendingTasks() const {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

void ThreadPool::WorkerLoop(std::stop_token stop_token) {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // condition_variable_any::wait() with stop_token:
            // Returns true if the predicate is satisfied.
            // Returns false (immediately) if stop was requested.
            bool got_work = cv_.wait(lock, stop_token, [this] {
                return !task_queue_.empty();
            });

            if (!got_work) {
                // Stop was requested. Drain any remaining tasks before exiting.
                while (!task_queue_.empty()) {
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                    lock.unlock();
                    task();
                    lock.lock();
                }
                return;
            }

            task = std::move(task_queue_.front());
            task_queue_.pop();
        }

        task();
    }
}
