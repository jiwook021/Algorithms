/**
 * @file test_CircularQueue.cpp
 * @brief Unit tests for CircularQueue and SafeQueue classes.
 *
 * Tests verify enqueue/dequeue, full/empty states, wrap-around behavior,
 * capacity reporting, and concurrent SafeQueue producer-consumer behavior.
 */

#include "CircularQueue.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <atomic>

/** @brief Verify basic enqueue and dequeue. */
static void TestEnqueueDequeue() {
    CircularQueue q(5);
    assert(q.Enqueue(10));
    assert(q.Enqueue(20));
    assert(q.Count() == 2u);

    int out = 0;
    assert(q.Dequeue(out));
    assert(out == 10);
    assert(q.Dequeue(out));
    assert(out == 20);
    assert(q.Count() == 0u);
    std::cout << "[PASS] TestEnqueueDequeue\n";
}

/** @brief Verify full queue rejects further enqueues. */
static void TestFullQueue() {
    CircularQueue q(3);
    assert(q.Enqueue(1));
    assert(q.Enqueue(2));
    assert(q.Enqueue(3));
    assert(q.Full());
    assert(!q.Enqueue(4));
    std::cout << "[PASS] TestFullQueue\n";
}

/** @brief Verify empty queue rejects dequeue. */
static void TestEmptyQueue() {
    CircularQueue q(3);
    assert(q.Empty());
    int out;
    assert(!q.Dequeue(out));
    std::cout << "[PASS] TestEmptyQueue\n";
}

/** @brief Verify wrap-around behavior. */
static void TestWrapAround() {
    CircularQueue q(3);
    q.Enqueue(1);
    q.Enqueue(2);
    q.Enqueue(3);

    int out;
    q.Dequeue(out); // removes 1
    q.Dequeue(out); // removes 2

    assert(q.Enqueue(4));
    assert(q.Enqueue(5));

    q.Dequeue(out); assert(out == 3);
    q.Dequeue(out); assert(out == 4);
    q.Dequeue(out); assert(out == 5);
    assert(q.Empty());
    std::cout << "[PASS] TestWrapAround\n";
}

/** @brief Verify Capacity accessor. */
static void TestCapacity() {
    CircularQueue q(7);
    assert(q.Capacity() == 7u);
    std::cout << "[PASS] TestCapacity\n";
}

/** @brief Verify SafeQueue concurrent produce/consume preserves sum. */
static void TestSafeQueueConcurrent() {
    SafeQueue q(5);
    std::atomic<int> sum{0};
    constexpr int N = 20;

    std::thread producer([&] {
        for (int i = 1; i <= N; ++i) q.Produce(i);
    });
    std::thread consumer([&] {
        for (int i = 0; i < N; ++i) {
            int out;
            if (q.Consume(out)) sum += out;
        }
    });

    producer.join();
    consumer.join();
    assert(sum.load() == N * (N + 1) / 2);
    std::cout << "[PASS] TestSafeQueueConcurrent\n";
}

int main() {
    TestEnqueueDequeue();
    TestFullQueue();
    TestEmptyQueue();
    TestWrapAround();
    TestCapacity();
    TestSafeQueueConcurrent();

    std::cout << "\nAll CircularQueue tests passed.\n";
    return 0;
}
