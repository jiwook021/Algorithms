/**
 * @file test_PriorityInversion.cpp
 * @brief Unit tests for PriorityInversion module.
 *
 * Tests the SharedResource mutex correctness under concurrent access.
 * Does not require real-time scheduling privileges.
 */

#include "PriorityInversion.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

/// Multiple threads increment a shared resource; final value must be exact.
void TestMutexProtectsResource() {
    PriorityInversion::SharedResource resource;
    constexpr int NUM_THREADS = 10;
    constexpr int INCREMENTS_PER_THREAD = 100;

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&resource]() {
            for (int j = 0; j < INCREMENTS_PER_THREAD; ++j) {
                resource.Acquire(1);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    assert(resource.GetValue() == NUM_THREADS * INCREMENTS_PER_THREAD);
    std::cout << "[PASS] TestMutexProtectsResource\n";
}

/// Two threads add known values; sum is deterministic.
void TestConcurrentAccess() {
    PriorityInversion::SharedResource resource;

    std::thread t1([&]() { resource.Acquire(5); });
    std::thread t2([&]() { resource.Acquire(10); });
    t1.join();
    t2.join();

    assert(resource.GetValue() == 15);
    std::cout << "[PASS] TestConcurrentAccess\n";
}

/// Single-threaded sanity check.
void TestSingleThread() {
    PriorityInversion::SharedResource resource;
    resource.Acquire(42);
    assert(resource.GetValue() == 42);
    std::cout << "[PASS] TestSingleThread\n";
}

/// Multiple acquire calls accumulate correctly.
void TestMultipleAcquires() {
    PriorityInversion::SharedResource resource;
    for (int i = 1; i <= 10; ++i) {
        resource.Acquire(i);
    }
    // 1+2+...+10 = 55
    assert(resource.GetValue() == 55);
    std::cout << "[PASS] TestMultipleAcquires\n";
}

int main() {
    TestMutexProtectsResource();
    TestConcurrentAccess();
    TestSingleThread();
    TestMultipleAcquires();
    std::cout << "\nAll PriorityInversion tests passed.\n";
    return 0;
}
