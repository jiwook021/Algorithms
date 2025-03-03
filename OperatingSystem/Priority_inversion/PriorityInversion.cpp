/**
 * @file PriorityInversion.cpp
 * @brief Implementation of priority inversion helpers and tasks.
 */

#include "PriorityInversion.hpp"

namespace PriorityInversion {

void BusyWork(int seconds) {
    auto endTime = std::chrono::steady_clock::now() +
                   std::chrono::seconds(seconds);
    while (std::chrono::steady_clock::now() < endTime) {
        std::this_thread::yield();
    }
}

void SetThreadPriority(pthread_t thread, int policy, int priority) {
    sched_param schParams{};
    schParams.sched_priority = priority;
    int rc = pthread_setschedparam(thread, policy, &schParams);
    CheckPthread(rc, "pthread_setschedparam");
}

void SharedResource::Acquire(int delta) {
    std::lock_guard<std::mutex> lock(mtx_);
    value_ += delta;
}

int SharedResource::GetValue() const {
    return value_;
}

// --- Static mutex for the POSIX demo tasks ---
static pthread_mutex_t piMutex;

void InitPiMutex() {
    pthread_mutexattr_t attr;
    CheckPthread(pthread_mutexattr_init(&attr), "mutexattr_init");
    CheckPthread(pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT),
                 "mutexattr_setprotocol");
    CheckPthread(pthread_mutex_init(&piMutex, &attr), "mutex_init");
    pthread_mutexattr_destroy(&attr);
}

void DestroyPiMutex() {
    pthread_mutex_destroy(&piMutex);
}

void* LowPriorityTask(void*) {
    std::cout << "[Low]  trying to lock mutex\n";
    pthread_mutex_lock(&piMutex);
    std::cout << "[Low]  locked mutex, performing CPU-bound work...\n";
    BusyWork(3);
    std::cout << "[Low]  unlocking mutex\n";
    pthread_mutex_unlock(&piMutex);
    return nullptr;
}

void* MediumPriorityTask(void*) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "[Med]  starting CPU-bound work\n";
    BusyWork(3);
    std::cout << "[Med]  done\n";
    return nullptr;
}

void* HighPriorityTask(void*) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "[High] trying to lock mutex\n";
    pthread_mutex_lock(&piMutex);
    std::cout << "[High] locked mutex\n";
    pthread_mutex_unlock(&piMutex);
    return nullptr;
}

} // namespace PriorityInversion
