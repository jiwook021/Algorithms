/**
 * @file PriorityInversion.hpp
 * @brief Priority inversion demonstration with priority inheritance protocol
 * @details Shows how a low-priority thread holding a mutex can block a
 *          high-priority thread while a medium-priority thread runs freely.
 *          The PTHREAD_PRIO_INHERIT protocol is used to mitigate this.
 */

#pragma once

#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>
#include <functional>

namespace PriorityInversion {

/// Check a pthread return code; throw on error.
inline void CheckPthread(int rc, const char* msg) {
    if (rc != 0) {
        std::cerr << msg << " failed: " << std::strerror(rc) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

/// Busy-loop for the given number of seconds (keeps thread runnable).
void BusyWork(int seconds);

/// Set thread scheduling policy and priority.
void SetThreadPriority(pthread_t thread, int policy, int priority);

/// Thread-safe shared resource for testability.
class SharedResource {
public:
    /// Acquire mutex and add delta to internal value.
    void Acquire(int delta);

    /// Return the current accumulated value.
    int GetValue() const;

private:
    std::mutex mtx_;
    int value_ = 0;
};

// POSIX thread entry points
void* LowPriorityTask(void* arg);
void* MediumPriorityTask(void* arg);
void* HighPriorityTask(void* arg);

} // namespace PriorityInversion
