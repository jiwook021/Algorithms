/**
 * @file WorkerThreadPool.cpp
 * @brief Implementation of WorkerThreadPool using std::jthread + stop_token.
 *
 * @details
 * Each worker uses condition_variable_any::wait() with its stop_token so that
 * a stop request unblocks the wait without an explicit atomic flag.  After
 * waking, the worker drains any remaining tasks before exiting, ensuring no
 * enqueued work is lost on shutdown.
 */

#include "WorkerThreadPool.hpp"
#include <utility>

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

WorkerThreadPool::WorkerThreadPool(std::size_t num_threads) {
    workers_.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this](std::stop_token st) { WorkerLoop(std::move(st)); });
    }
}

WorkerThreadPool::~WorkerThreadPool() {
    Shutdown();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void WorkerThreadPool::EnqueueTask(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
    }
    cv_.notify_one();
}

void WorkerThreadPool::Shutdown() {
    // Request stop on every worker jthread.  Each jthread's stop_source is
    // independent, so we iterate and request individually.
    for (auto& w : workers_) {
        w.request_stop();
    }
    cv_.notify_all();

    // jthread destructor joins automatically, but we join explicitly here so
    // that Shutdown() blocks until all workers have finished.
    for (auto& w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
}

std::size_t WorkerThreadPool::ThreadCount() const {
    return workers_.size();
}

bool WorkerThreadPool::IsRunning() const {
    // The pool is running if no worker has received a stop request yet.
    // We check the first worker; if it has been stopped, all have been.
    if (workers_.empty()) {
        return false;
    }
    return !workers_.front().get_stop_token().stop_requested();
}

std::size_t WorkerThreadPool::PendingTasks() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

void WorkerThreadPool::WorkerLoop(std::stop_token stop_token) {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // condition_variable_any::wait() with stop_token:
            //   - returns true  if the predicate is satisfied (task available).
            //   - returns false if a stop was requested.
            bool has_task = cv_.wait(lock, stop_token, [this] {
                return !task_queue_.empty();
            });

            if (!has_task) {
                // Stop was requested.  Drain any remaining tasks before exiting.
                while (!task_queue_.empty()) {
                    auto remaining = std::move(task_queue_.front());
                    task_queue_.pop();
                    lock.unlock();
                    try {
                        remaining();
                    } catch (const std::exception& e) {
                        std::cerr << "Exception during drain: " << e.what()
                                  << std::endl;
                    }
                    lock.lock();
                }
                return;
            }

            task = std::move(task_queue_.front());
            task_queue_.pop();
        }

        try {
            task();
        } catch (const std::exception& e) {
            std::cerr << "Exception caught during task execution: "
                      << e.what() << std::endl;
        }
    }
}
